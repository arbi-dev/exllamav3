from __future__ import annotations
import torch
from ...model.config import Config
from .exl3_lib.quantize import preapply_had_l, preapply_had_r, had_k, had_n
from ...ext import exllamav3_ext as ext
from ...util.tensor import g_tensor_cache
import os
from ...util import profile_opt

# Rows above which forward() dequantizes the weight once and runs cuBLAS
# instead of tiling the trellis GEMM over M. The trellis kernel reads the
# packed weight once per M tile and runs the tiles serially, so leg A costs
# ceil(rows / TILESIZE_M) passes while reconstruct+hgemm costs exactly one at
# any row count; the threshold is where the second is cheaper.
#
# It moved 144 -> 280 with the multi-row-block shape table. At the old
# four-shape table every tile was 16 rows deep, and 144 was well calibrated
# for it: the census-weighted crossover over a 27B EXL3 checkpoint measured
# 165 there. The table now reaches 64 rows, which cuts leg A's pass count by
# 4x in the deepest tile and moves the same measured crossover to 325. 280
# rather than 325 because the cost curve is not symmetric about the crossing —
# leg A's cost steps with the tile while leg B's is smooth, so being early to
# leg B costs less than being late — and because 280 captures 99.4% of what
# any single constant can recover across the row counts a prefill chunk
# actually takes.
AUTO_RECONSTRUCT_THRESHOLD = 280

# A separate, much lower threshold for NARROW OUTPUT WIDTHS, where the trellis
# GEMM stops being bandwidth-bound and starts being occupancy-bound. Leg A
# tiles N at 128 at minimum, so an out_features of 1024 gives it 8 column
# blocks to fill the machine with; measured on an RTX 4090 (128 SMs) a
# 5120x1024 linear runs 8x off its own roofline at 32 rows, where a 5120x17408
# one is within 1.8x. Leg B has no such floor — its reconstruct is a plain
# memory-bound write and cuBLAS tiles the GEMM itself — so it takes over far
# earlier: crossover measured between 32 and 40 rows for 5120x1024, against
# 230-427 for every wider geometry in the same checkpoint.
#
# The bound is a multiple of 128, so it reads the same on a Hadamard-padded
# out_features as on the declared one (padding rounds up to 128, and anything
# at or below 2048 stays at or below it) — the gate cannot disagree between
# this dispatch and a caller that mirrors it from kernel dims.
#
# 64 rather than the measured ~38 crossing for two reasons that agree: it is
# the deepest tile the shape table offers, so leg A keeps its whole cheapest
# regime; and it sits above the widest speculative verify slate any shipped
# recipe produces (max_batch 4 x K+1 8 = 32 rows), so a verify block and a
# plain decode of the same tokens never land on different legs. The rows
# between 38 and 64 are worth ~0.14 ms per forward across the 34 narrow
# linears of a 27B checkpoint, which is not worth buying either of those.
NARROW_RECONSTRUCT_MAX_N = 2048
NARROW_RECONSTRUCT_THRESHOLD = 64

MAX_RECONSTRUCT_SLICE_N = 32768
RECONSTRUCT_SLICE_GRANULARITY_N = 128


def auto_reconstruct_threshold(out_features: int) -> int:
    """Row count above which this linear should reconstruct rather than tile.

    One function rather than a bare constant because the crossover is a
    property of the SHAPE, not of the model: see NARROW_RECONSTRUCT_MAX_N.
    Callers that mirror this dispatch (arbi-serve's custom op does, to own its
    output buffer under cudagraph capture) must call this and never re-derive
    it, or the two implementations drift on a checkpoint neither was tested on.
    """
    if out_features <= NARROW_RECONSTRUCT_MAX_N:
        return NARROW_RECONSTRUCT_THRESHOLD
    return AUTO_RECONSTRUCT_THRESHOLD


# Both kernel axes must be whole Hadamard blocks or reconstruct_had_slice
# refuses the shape. EXL3 tensors are had-transformed on both sides at quant
# time so this always holds, and TP shards cut on whole blocks, but the gate
# below is mirrored by out-of-tree callers on the PADDED kernel dims, so the
# width has a name rather than being spelled 128 at each use.
FUSED_RECONSTRUCT_BLOCK = 128

# THE FUSED RECONSTRUCT'S ROW GATE, ONE VALUE PER OUTPUT-WIDTH CLASS.
#
# reconstruct_had_slice emits the dequantized weight already rotated by both
# 128-wide Hadamards and scaled by suh/svh, so the gemm consumes the raw
# activation and produces the final output: the two standalone had_r_128
# launches and the activation-sized scratch between them disappear. The saving
# is activation traffic, which scales with rows; the extra cost is weight-
# shaped, which does not. Hence a row threshold.
#
# WHAT THE OLD SINGLE CONSTANT (1024) RESTED ON, AND WHY IT IS WRONG.
# It was justified here as "the fused kernel costs ~4x plain reconstruct
# (k*n-proportional) while the saved had launches scale with rows*(k+n);
# breakeven is rows ~400-900 across shapes". The first half is false, and it is
# the half that sets the shape of the answer. Timed with the two reconstruct
# kernels ALONE, no activation in the frame, round-robin and L2-flushed against
# a null control (arbi-serve tools/exl3_largem_dispatch_bench.py
# --reconstruct-probe, arbicity/arbi-serve#1755, 1x RTX 4090, 4.0bpw 27B):
#
#     5120x1024    0.0144 -> 0.0236 ms   1.64x   +9.2 us
#     6144x5120    0.0911 -> 0.0973      1.07x   +6.1
#     17408x5120   0.2509 -> 0.2580      1.03x   +7.2
#     5120x10240   0.1505 -> 0.1577      1.05x   +7.2
#     5120x12288   0.1784 -> 0.1925      1.08x   +14.1
#     5120x17408   0.2518 -> 0.2632      1.05x   +11.4
#     5120x248320  3.8765 -> 3.9700      1.02x   +93.6  (8 slices, ~12 us each)
#
# Not 4x, and not proportional to k*n: 6-14 us PER LAUNCH across a 17x range of
# weight sizes. A near-FIXED cost against a saving LINEAR in rows crosses far
# earlier than a k*n-proportional one, and it crosses at a different row count
# for a shape that pays the cost once per slice than for one that pays it once.
# One scalar cannot express that, which is why this is a function of the output
# width -- keyed on the same two bounds auto_reconstruct_threshold already
# splits on, so a caller mirroring both gates asks one question about a shape
# and not two. Both bounds are multiples of FUSED_RECONSTRUCT_BLOCK, so they
# read the same on a Hadamard-padded out_features as on the declared one.
#
# The values are the LOWEST ROW COUNT AT WHICH THE SWEEP OBSERVED THE FUSED
# VARIANT WINNING, never an interpolated crossing: both variants driven through
# the real dispatch with only this gate moved, a kernel-counting pass
# certifying which variant each row count reached before its timing is trusted,
# paired per-round deltas with a sign-agreement count, against a null control
# that is the standalone arm run twice (--fused-crossover, same PR).

# WIDE, SINGLE-SLICE -- 5120x17408, 17408x5120, 6144x5120, 5120x10240,
# 5120x12288, i.e. 95% of a prefill chunk's linear FLOPs. THE GATE HAS NO
# CORRECT POSITIVE VALUE HERE: the fused variant already wins at 64 rows on the
# three widest and by 128 rows on the rest, while auto_reconstruct_threshold
# does not admit this leg at all below 281 rows. The crossover is below the row
# count at which the leg is reachable, so the gate is not conservative, it is
# only ever wrong -- over 281-1023 rows the old constant gave up 2.3-9.0% with
# an fp32 cuBLAS accumulator and 3.3-14.5% with fp16. 0 also deletes a
# row-keyed algorithm switch from the class that carries almost all of prefill,
# which is a determinism property and not only a speed one.
WIDE_FUSED_RECONSTRUCT_THRESHOLD = 0

# NARROW (out_features <= NARROW_RECONSTRUCT_MAX_N) -- 5120x1024, the k_proj /
# v_proj that write the KV. The one class where a row gate on this leg earns
# its existence: the fused variant LOSES here below the crossing, by up to 18%
# at 65-96 rows, because a 1024-wide weight makes the per-launch cost a large
# fraction of the whole reconstruct (the 1.64x row of the probe above). The
# crossing brackets 400-500; 512 is the lowest row count at which the fused
# variant was measured to win rather than interpolated to (+9.0% fp32 / +7.4%
# fp16, rising to +19.3% / +20.7% by 1023 rows). The bracket's interior is not
# taken because the error is asymmetric on exactly this class -- early is a
# measured loss on the linears whose output is the KV, late is a bounded
# forfeit over 400-512 rows.
NARROW_FUSED_RECONSTRUCT_THRESHOLD = 512

# WIDE, N-SLICED (out_features > MAX_RECONSTRUCT_SLICE_N) -- the 5120x248320
# head, 8 slices. Same per-launch cost as the wide single-slice class but paid
# once PER SLICE, so it crosses later: bracket ~300-350, and 384 is the lowest
# measured win (+0.6%, rising to +5.2% by 1023 rows). The gains are small
# because the head's gemm dominates its own reconstruct; the value is here so
# the class is decided by its own measurement rather than by the wide class's.
SLICED_FUSED_RECONSTRUCT_THRESHOLD = 384


def fused_reconstruct_threshold(out_features: int) -> int:
    """Rows at or above which this linear should take the FUSED reconstruct.

    A function of the output width for the same reason
    auto_reconstruct_threshold is one, and split on the same two bounds: the
    cost this trades away is per-kernel-launch, so it lands differently on a
    narrow weight (where it is a large fraction of the reconstruct), on a wide
    one (where it is noise), and on one wide enough to be sliced (where it is
    paid once per slice). See the measurements above each value.

    ``out_features`` may be the declared width or the Hadamard-padded kernel
    width; both bounds are multiples of FUSED_RECONSTRUCT_BLOCK, so the two
    cannot disagree. Callers that mirror this dispatch (arbi-serve's custom op
    does, to own its output buffer under cudagraph capture) must call this and
    never restate it -- a divergence here is a different ALGORITHM on the same
    rows, not a rounding difference.
    """
    if out_features <= NARROW_RECONSTRUCT_MAX_N:
        return NARROW_FUSED_RECONSTRUCT_THRESHOLD
    if out_features > MAX_RECONSTRUCT_SLICE_N:
        return SLICED_FUSED_RECONSTRUCT_THRESHOLD
    return WIDE_FUSED_RECONSTRUCT_THRESHOLD


no_fused_reconstruct = os.environ.get("EXL3_NO_FUSED_RECONSTRUCT", "0") != "0"

class LinearEXL3:

    quant_type: str = "exl3"

    def __init__(
        self,
        config: Config | None,
        in_features: int,
        out_features: int,
        scale: torch.Tensor | None = None,
        su: torch.Tensor | None = None,
        sv: torch.Tensor | None = None,
        suh: torch.Tensor | None = None,
        svh: torch.Tensor | None = None,
        trellis: torch.Tensor | None = None,
        mcg: torch.Tensor | None = None,
        mul1: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        transformers_fix: bool = False,
        key: str | None = None
    ):
        assert scale is None, "scale is no longer used"
        assert su is not None or suh is not None, "either su (packed) or suh (unpacked) is required"
        assert sv is not None or svh is not None, "either sv (packed) or svh (unpacked) is required"
        assert trellis is not None, "trellis is required"
        if su is not None: assert su.dtype == torch.int16, "su is wrong datatype"
        if sv is not None: assert sv.dtype == torch.int16, "sv is wrong datatype"
        if suh is not None: assert suh.dtype == torch.half, "suh is wrong datatype"
        if svh is not None: assert svh.dtype == torch.half, "svh is wrong datatype"
        assert trellis.dtype == torch.int16, "trellis is wrong datatype"
        assert len(trellis.shape) == 3, "trellis must have dim = 3"

        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)

        # Not a Module subclass, so the config-or-NullConfig default doesn't apply here; TP imports pass
        # config=None and forward() reads config.infer_params
        if config is None:
            from ...model.config import NullConfig
            config = NullConfig()
        self.config = config
        self.transformers_fix = transformers_fix
        self.key = key

        # self.scale = scale.item()
        self.su = None
        self.sv = None
        self.suh = suh if suh is not None else self.unpack_bf(su)
        self.svh = svh if svh is not None else self.unpack_bf(sv)
        self.trellis = trellis
        self.K = trellis.shape[-1] // 16
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.out_dtype = out_dtype
        self.default_out_dtype = out_dtype or torch.half

        self.mcg_tensor = mcg
        self.mul1_tensor = mul1
        self.mcg = self.mcg_tensor is not None
        self.mul1 = self.mul1_tensor is not None

        self._fused_reconstruct = None
        self.bsz1_xh_args = (self.trellis.device, (1, self.in_features), self.out_dtype)
        self.bc = ext.BC_LinearEXL3(
            self.trellis,
            self.suh,
            self.svh,
            self.K,
            self.bias,
            self.mcg,
            self.mul1,
            g_tensor_cache.get(*self.bsz1_xh_args)
        )


    def unload(self):
        # g_tensor_cache.drop(*self.bsz1_xh_args)
        pass


    def get_tensors(self, key: str):
        return {
            f"{key}.{subkey}": tensor.contiguous()
            for subkey, tensor in [
                ("su", self.su),
                ("sv", self.sv),
                ("suh", self.suh),
                ("svh", self.svh),
                ("trellis", self.trellis),
                ("bias", self.bias),
                ("mcg", self.mcg_tensor),
                ("mul1", self.mul1_tensor),
            ] if tensor is not None
        }


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if "ovr" in params:
            ovr = params["ovr"]
            if self.key in ovr and ovr[self.key].inner is not self:
                return ovr[self.key].forward(x, params, out_dtype)

        # The EXL3 kernels read x as contiguous rows; a strided view (e.g. a head-group slice
        # of a wider tensor) would be silently misread as interleaved garbage. Producers are
        # responsible for contiguity (a silent copy here would hide a hot-path inefficiency
        # and break CUDA-graph address stability)
        assert x.is_contiguous(), f"LinearEXL3 {self.key}: non-contiguous input {tuple(x.shape)}"

        reconstruct = params.get("reconstruct")
        if not reconstruct:
            rows = x.numel() // x.shape[-1]
            if rows <= auto_reconstruct_threshold(self.out_features) \
                    or self.config.infer_params.no_reconstruct:
                dtype = out_dtype or self.default_out_dtype
                return self.bc.run_alloc(x, self.out_features, dtype == torch.float)

        return self.reconstruct_hgemm(x, out_dtype)


    def unpack_bf(self, bitfield: torch.Tensor):
        # For some reason this operation causes a GPU assert on Transformers. Running on CPU seems to fix it
        device = bitfield.device
        if self.transformers_fix:
            bitfield = bitfield.cpu()

        # (Only used for full reconstruct and loading old models, not during inference)
        bitfield = bitfield.view(torch.uint16).to(torch.int)
        masks = (1 << torch.arange(16)).to(bitfield.device)
        expanded = (bitfield.unsqueeze(-1) & masks) > 0
        expanded = expanded.flatten()
        # NOT torch.where with CPU scalar tensors: that path misses the device guard when the
        # condition lives on a non-current device (observed on torch 2.11.0+cu130) — the kernel
        # launches on the current device's context, faults there, silently zero-fills the output
        # and leaves every other device in the process unusable. Map bool -> {-1, +1} arithmetically
        expanded = 1.0 - expanded.to(torch.float16) * 2.0
        return expanded.contiguous().to(device)


    def reconstruct_hgemm(self, x: torch.Tensor, out_dtype):

        shape = x.shape
        rows = x.numel() // shape[-1]
        out_shape = shape[:-1] + (self.out_features,)
        x = x.view(rows, self.in_features)
        y = torch.empty(out_shape, dtype = out_dtype or self.default_out_dtype, device = x.device)

        y_ = y.view(rows, self.out_features)

        # Fused path: reconstruct emits ORIGINAL-basis weights (both Hadamards + sign
        # vectors folded into the memory-bound reconstruct kernel), so the gemm runs on the
        # raw input and the standalone input/output had_r_128 launches disappear (~14% of
        # long-chunk prefill GPU time). Requires 128-divisible dims (always true for EXL3
        # tensors: both sides are had-transformed at quant time)
        if self._fused_reconstruct is None:
            self._fused_reconstruct = (
                self.in_features % FUSED_RECONSTRUCT_BLOCK == 0
                and self.out_features % FUSED_RECONSTRUCT_BLOCK == 0
                and not no_fused_reconstruct
            )

        # Row gate, per output-width class — see fused_reconstruct_threshold
        use_fused = self._fused_reconstruct and rows >= fused_reconstruct_threshold(self.out_features)

        if use_fused:
            xh = x
        else:
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, self.suh, None, 1.0)

        if self.out_features <= MAX_RECONSTRUCT_SLICE_N:
            w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
            if use_fused:
                ext.reconstruct_had_slice(w, self.trellis, self.suh, self.svh, self.K, self.mcg, self.mul1, 0)
            else:
                ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
            ext.hgemm(xh, w, y_)
        else:
            numel_ = self.in_features * MAX_RECONSTRUCT_SLICE_N
            w_ = torch.empty((numel_,), dtype = torch.half, device = self.trellis.device)
            for n_start in range(0, self.out_features, MAX_RECONSTRUCT_SLICE_N):
                n_end = min(n_start + MAX_RECONSTRUCT_SLICE_N, self.out_features)
                numel = self.in_features * (n_end - n_start)
                w = w_[:numel].view(self.in_features, n_end - n_start)
                if use_fused:
                    ext.reconstruct_had_slice(
                        w, self.trellis, self.suh, self.svh[n_start:], self.K, self.mcg, self.mul1, n_start)
                else:
                    ext.reconstruct_slice(w, self.trellis, self.K, self.mcg, self.mul1, n_start)
                ext.hgemm(xh, w, y_[:, n_start:n_end])

        if not use_fused:
            ext.had_r_128(y_, y_, None, self.svh, 1.0)

        if self.bias is not None:
            y += self.bias
        return y


    def get_inner_weight_tensor(self, n_offset: int = 0, n_features: int | None = None):
        w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
        ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
        return w


    def get_weight_tensor(self):
        # suh = self.unpack_bf(self.su).unsqueeze(1)
        suh = self.unpack_bf(self.su).unsqueeze(1) if self.su else self.suh.unsqueeze(1)
        svh = self.unpack_bf(self.sv).unsqueeze(0) if self.sv else self.svh.unsqueeze(0)
        w = self.get_inner_weight_tensor()
        w = preapply_had_l(w, had_k)
        w *= suh
        w = preapply_had_r(w, had_n)
        w *= svh
        # w *= self.scale
        return w


    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias


    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.trellis.device
        if self.su is not None: self.su = self.su.cpu()
        if self.sv is not None: self.sv = self.sv.cpu()
        if self.suh is not None: self.suh = self.suh.cpu()
        if self.svh is not None: self.svh = self.svh.cpu()
        if self.trellis is not None: self.trellis = self.trellis.cpu()
        if self.bias is not None: self.bias = self.bias.cpu()


    def unswap_cpu(self):
        if self.swap_device is None:
            return
        if self.su is not None: self.su = self.su.to(self.swap_device)
        if self.sv is not None: self.sv = self.sv.to(self.swap_device)
        if self.suh is not None: self.suh = self.suh.to(self.swap_device)
        if self.svh is not None: self.svh = self.svh.to(self.swap_device)
        if self.trellis is not None: self.trellis = self.trellis.to(self.swap_device)
        if self.bias is not None: self.bias = self.bias.to(self.swap_device)
        self.swap_device = None


    def tp_export(self, plan, producer):
        return {
            "cls": LinearEXL3,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "suh": producer.send(self.suh),
            "svh": producer.send(self.svh),
            "trellis": producer.send(self.trellis),
            "bias": producer.send(self.bias),
            "mcg": producer.send(self.mcg_tensor),
            "mul1": producer.send(self.mul1_tensor),
            "out_dtype": self.out_dtype,
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            suh = consumer.recv(id_suh, cuda = True, slice_dim = 0, first = first, last = last)
            svh = consumer.recv(id_svh, cuda = True)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 0, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True) if first == 0 else None
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module


    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2, dbg = False):
        return LinearEXL3.tp_import_split_n(local_context, exported, plan, [split_0, split_1, split_2], dbg)


    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)

        svh_ = []
        trellis_ = []
        bias_ = []
        in_features = 0
        out_features = 0

        for split in splits:
            assert split is not None
            split_out, first, last = split
            assert split_out

            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features += last - first
            svh_.append(svh)
            trellis_.append(trellis)
            bias_.append(bias)

        svh = torch.cat(svh_, dim = 0)
        trellis = torch.cat(trellis_, dim = 1)
        bias = torch.cat(bias_, dim = 0) if bias_[0] is not None else None

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module
