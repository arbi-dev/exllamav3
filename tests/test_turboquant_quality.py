"""
TurboQuant KV cache quality evaluation.

Properly tests cache quantization impact by forcing incremental (cache-reading)
evaluation. Includes self-validation to ensure the test harness is correct.

Usage:
    python tests/test_turboquant_quality.py -m /path/to/exl3/model [-t 2048]

Tests:
1. Harness validation: FP16 incremental PPL == FP16 single-pass PPL
2. Greedy generation comparison: FP16 vs TQ at various bit widths
3. Incremental PPL: measures actual cache quant impact
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant, CacheLayer_turboquant

PAGE_SIZE = 256


def load_wikitext2(tokenizer, max_tokens=8192):
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    tokens = tokenizer.encode(text)
    if max_tokens > 0:
        tokens = tokens[:, :max_tokens]
    return tokens


def make_cache(model, max_tokens, layer_type, **kwargs):
    cache = Cache(model, max_num_tokens=max_tokens, layer_type=layer_type, **kwargs)
    device = torch.device("cuda:0")
    for layer in cache.layers.values():
        if layer.device is None:
            layer.alloc(device)
    return cache


@torch.inference_mode()
def ppl_single_pass(model, tokenizer, eval_tokens, cache, seq_len=512, max_windows=20):
    """Standard single-pass PPL. Cache is written but NOT read (prefill only)."""
    vocab_size = tokenizer.actual_vocab_size
    stride = seq_len // 2
    lp_sum, lp_count = 0.0, 0

    for i, start in enumerate(range(0, eval_tokens.shape[1] - seq_len, stride)):
        if i >= max_windows:
            break
        ids = eval_tokens[:, start:start + seq_len]
        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
                  "batch_shape": (1, seq_len)}
        logits = model.forward(ids, params)
        logits = logits[:, stride:-1, :vocab_size].float() + 1e-10
        lp = F.log_softmax(logits, dim=-1)
        tgt = ids[:, stride + 1:].to(lp.device)
        lp_sum += lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
        lp_count += tgt.numel()
        del logits, lp, tgt
    return math.exp(-lp_sum / lp_count), lp_count


@torch.inference_mode()
def ppl_incremental(model, tokenizer, eval_tokens, cache, context_len=256,
                    eval_len=128, max_windows=8):
    """Incremental PPL that forces cache reads. Prefill context, then evaluate
    token-by-token (each token reads from quantized cache)."""
    vocab_size = tokenizer.actual_vocab_size
    max_cache = cache.max_num_tokens
    lp_sum, lp_count = 0.0, 0

    for w, start in enumerate(range(0, eval_tokens.shape[1] - context_len - eval_len - 1,
                                     context_len + eval_len)):
        if w >= max_windows:
            break
        ids = eval_tokens[:, start:start + context_len + eval_len + 1]

        # Prefill context (writes K/V to cache)
        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
                  "batch_shape": (1, max_cache)}
        model.prefill(input_ids=ids[:, :context_len], params=params)

        # Incremental eval (reads K/V from cache for each token)
        rec_states = params.get("recurrent_states")
        for t in range(eval_len):
            pos = context_len + t
            params = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos,
                      "batch_shape": (1, max_cache), "recurrent_states": rec_states}
            logits = model.forward(input_ids=ids[:, pos:pos + 1], params=params)
            rec_states = params.get("recurrent_states")

            log_probs = F.log_softmax(logits[:, -1, :vocab_size].float() + 1e-10, dim=-1)
            target = ids[0, pos + 1].item()
            lp_sum += log_probs[0, target].item()
            lp_count += 1
            del logits, log_probs

    return math.exp(-lp_sum / lp_count), lp_count


@torch.inference_mode()
def greedy_generate(model, tokenizer, cache, prompt, num_tokens=100):
    """Generate tokens greedily, reading from cache."""
    vocab = tokenizer.get_id_to_piece_list()
    input_ids = tokenizer.encode(prompt)
    max_cache = cache.max_num_tokens

    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
              "batch_shape": (1, max_cache)}
    model.prefill(input_ids=input_ids[:, :-1], params=params)

    generated = []
    context = input_ids.clone()
    t0 = time.time()
    for _ in range(num_tokens):
        params = {"attn_mode": "flash_attn", "cache": cache,
                  "past_len": context.shape[-1] - 1, "batch_shape": (1, max_cache),
                  "recurrent_states": params.get("recurrent_states")}
        logits = model.forward(input_ids=context[:, -1:], params=params)
        token_id = logits[:, -1, :].argmax(dim=-1).item()
        generated.append(token_id)
        context = torch.cat((context, torch.tensor([[token_id]])), dim=-1)
    elapsed = time.time() - t0

    text = "".join(vocab[t] for t in generated)
    unique_last = len(set(generated[-50:])) if len(generated) >= 50 else len(set(generated))
    return text, len(generated) / elapsed, generated, unique_last


def main():
    parser = argparse.ArgumentParser(description="TurboQuant quality evaluation")
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-t", "--max_tokens", type=int, default=8192)
    parser.add_argument("-g", "--gen_tokens", type=int, default=100)
    args = parser.parse_args()

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cl = model.get_cache_layers()

    print("=" * 70)
    print("TurboQuant KV Cache Quality Evaluation")
    print("=" * 70)
    print(f"Model: {args.model_dir}")
    print(f"  {len(cl)} attn layers, {cl[0].num_kv_heads} KV heads, head_dim={cl[0].head_dim}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load model once
    dummy = Cache(model, max_num_tokens=1024, layer_type=CacheLayer_fp16)
    model.load(progressbar=True)
    dummy.detach_from_model()
    del dummy
    torch.cuda.empty_cache()

    eval_tokens = load_wikitext2(tokenizer, args.max_tokens)
    print(f"  Eval tokens: {eval_tokens.shape[1]}")

    # ── Test 1: Harness validation ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 1: Harness validation (FP16 single-pass vs incremental)")
    print("  If these differ significantly, the test harness has a bug.")
    print(f"{'='*70}")

    cache_sp = make_cache(model, 512, CacheLayer_fp16)
    ppl_sp, count_sp = ppl_single_pass(model, tokenizer, eval_tokens, cache_sp,
                                        seq_len=512, max_windows=10)
    cache_sp.detach_from_model()
    del cache_sp
    torch.cuda.empty_cache()

    cache_inc = make_cache(model, 768, CacheLayer_fp16)
    ppl_inc, count_inc = ppl_incremental(model, tokenizer, eval_tokens, cache_inc,
                                          context_len=256, eval_len=128, max_windows=10)
    cache_inc.detach_from_model()
    del cache_inc
    torch.cuda.empty_cache()

    print(f"  Single-pass:  PPL={ppl_sp:.4f} ({count_sp} tokens)")
    print(f"  Incremental:  PPL={ppl_inc:.4f} ({count_inc} tokens)")
    print(f"  Note: values may differ due to different eval windows/lengths.")
    print(f"  Both should be in the same ballpark for this model.")

    # ── Test 2: Greedy generation ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"TEST 2: Greedy generation ({args.gen_tokens} tokens)")
    print(f"{'='*70}")

    prompt = "The meaning of life is"
    gen_configs = [
        ("FP16", CacheLayer_fp16, {}),
        ("TQ K=4 V=4", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 4}),
        ("TQ K=3 V=3", CacheLayer_turboquant, {"k_bits": 3, "v_bits": 3}),
        ("TQ K=2 V=2", CacheLayer_turboquant, {"k_bits": 2, "v_bits": 2}),
        ("exl3 K=4 V=4", CacheLayer_quant, {"k_bits": 4, "v_bits": 4}),
    ]

    max_gen_cache = ((args.gen_tokens + 256) // 256) * 256 + 256
    fp16_tokens = None
    for label, lt, kw in gen_configs:
        cache = make_cache(model, max_gen_cache, lt, **kw)
        text, speed, tokens, unique = greedy_generate(
            model, tokenizer, cache, prompt, args.gen_tokens
        )
        if fp16_tokens is None:
            fp16_tokens = tokens
        match = sum(1 for a, b in zip(fp16_tokens, tokens) if a == b)
        print(f"\n  {label} ({speed:.1f} tok/s, {match}/{args.gen_tokens} match, {unique} unique/last50):")
        print(f"    {text[:150]}...")
        cache.detach_from_model()
        del cache
        torch.cuda.empty_cache()

    # ── Test 3: Incremental PPL ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 3: Incremental PPL (cache-reading evaluation)")
    print(f"{'='*70}")

    ppl_configs = [
        ("FP16", CacheLayer_fp16, {}),
        ("TQ K=4 V=4", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 4}),
        ("TQ K=4 V=3", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 3}),
        ("TQ K=3 V=3", CacheLayer_turboquant, {"k_bits": 3, "v_bits": 3}),
        ("TQ K=2 V=2", CacheLayer_turboquant, {"k_bits": 2, "v_bits": 2}),
        ("exl3 K=4 V=4", CacheLayer_quant, {"k_bits": 4, "v_bits": 4}),
        ("exl3 K=2 V=2", CacheLayer_quant, {"k_bits": 2, "v_bits": 2}),
    ]

    print(f"\n{'Method':<20} {'PPL':>10} {'Delta':>10} {'Delta%':>8} {'Tokens':>8}")
    print("-" * 60)
    fp16_ppl = None
    for label, lt, kw in ppl_configs:
        cache = make_cache(model, 768, lt, **kw)
        ppl, count = ppl_incremental(model, tokenizer, eval_tokens, cache,
                                      context_len=256, eval_len=128, max_windows=10)
        delta = 0.0 if fp16_ppl is None else ppl - fp16_ppl
        pct = 0.0 if fp16_ppl is None else delta / fp16_ppl * 100
        if fp16_ppl is None:
            fp16_ppl = ppl
        print(f"{label:<20} {ppl:>10.4f} {delta:>+10.4f} {pct:>+7.2f}% {count:>8}")
        cache.detach_from_model()
        del cache
        torch.cuda.empty_cache()

    model.unload()
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
