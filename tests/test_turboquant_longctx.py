"""
Comprehensive TurboQuant tests: long context PPL, asymmetric K/V, stress test.
Model loaded ONCE, caches swapped efficiently.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
PAGE_SIZE = 256
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant, CacheLayer_turboquant

MODEL_DIR = "/mnt/k8scache/models/Qwen3.5-27B-exl3-4.00bpw"


def get_wikitext2_tokens(tokenizer, max_tokens=None):
    dataset_text = "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    tokens = tokenizer.encode(dataset_text)
    if max_tokens:
        tokens = tokens[:, :max_tokens]
    return tokens


@torch.inference_mode()
def eval_ppl(model, cache, tokenizer, eval_tokens, seq_len, max_windows=20):
    vocab_size = tokenizer.actual_vocab_size
    num_tokens = eval_tokens.shape[1]
    stride = seq_len // 2
    logprob_sum = 0.0
    logprob_count = 0

    for i, start in enumerate(range(0, num_tokens - seq_len, stride)):
        if i >= max_windows:
            break
        input_ids = eval_tokens[:, start:start + seq_len]
        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, seq_len)}
        logits = model.forward(input_ids, params)
        eval_start = max(0, seq_len - stride)
        logits = logits[:, eval_start:-1, :vocab_size].float() + 1e-10
        log_probs = F.log_softmax(logits, dim=-1)
        target_ids = input_ids[:, eval_start + 1:].to(log_probs.device)
        logprob_sum += log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).sum().item()
        logprob_count += target_ids.numel()
        del logits, log_probs, target_ids
    torch.cuda.empty_cache()
    return math.exp(-logprob_sum / logprob_count), logprob_count


@torch.inference_mode()
def generate_tokens(model, cache, tokenizer, num_tokens=1000):
    vocab = tokenizer.get_id_to_piece_list()
    prompt = "Once upon a time in a land far away, there lived a wise old wizard who"
    input_ids = tokenizer.encode(prompt)
    max_len = ((num_tokens + input_ids.shape[1] + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE

    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, max_len)}
    model.prefill(input_ids=input_ids[:, :-1], params=params)

    generated = []
    context = input_ids.clone()
    t0 = time.time()
    for _ in range(num_tokens):
        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": context.shape[-1] - 1,
                  "batch_shape": (1, max_len), "recurrent_states": params.get("recurrent_states")}
        logits = model.forward(input_ids=context[:, -1:], params=params)
        token_id = logits[:, -1, :].argmax(dim=-1).item()
        generated.append(token_id)
        context = torch.cat((context, torch.tensor([[token_id]])), dim=-1)
    t1 = time.time()

    text = "".join(vocab[t] for t in generated)
    unique_last = len(set(generated[-100:]))
    return text, num_tokens / (t1 - t0), unique_last


def make_cache(model, max_tokens, layer_type, **kwargs):
    """Create cache on already-loaded model. Manually allocate layers."""
    cache = Cache(model, max_num_tokens=max_tokens, layer_type=layer_type, **kwargs)
    # If model already loaded, alloc wasn't called on the new cache layers
    device = torch.device("cuda:0")
    for layer in cache.layers.values():
        if layer.device is None:
            layer.alloc(device)
    return cache


def run_with_cache(model, tokenizer, eval_tokens, label, layer_type, cache_kwargs,
                   seq_len=1024, max_windows=20):
    """Create cache, run PPL eval, detach cache. Model stays loaded."""
    cache = make_cache(model, seq_len, layer_type, **cache_kwargs)
    ppl, count = eval_ppl(model, cache, tokenizer, eval_tokens, seq_len, max_windows)
    cache.detach_from_model()
    del cache
    torch.cuda.empty_cache()
    return ppl, count


def main():
    print("=" * 70)
    print("TurboQuant Comprehensive Tests")
    print("=" * 70)

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)

    print(f"Model: {MODEL_DIR}")
    cl = model.get_cache_layers()
    print(f"  {len(cl)} attn layers, {cl[0].num_kv_heads} KV heads, head_dim={cl[0].head_dim}")

    # Load model ONCE
    # We need the largest cache size we'll use. Create a dummy FP16 cache for loading.
    dummy_cache = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_fp16)
    t0 = time.time()
    model.load(progressbar=True)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    eval_tokens = get_wikitext2_tokens(tokenizer, max_tokens=65536)
    print(f"  Eval tokens: {eval_tokens.shape[1]}")

    # Detach dummy cache - we'll create specific caches for each test
    dummy_cache.detach_from_model()
    del dummy_cache
    torch.cuda.empty_cache()

    # ── TEST 1: PPL vs context length ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 1: Perplexity vs context length")
    print(f"{'='*70}")

    configs = [
        ("FP16", CacheLayer_fp16, {}),
        ("TQ K=4 V=4", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 4}),
        ("TQ K=4 V=3", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 3}),
        ("exl3 K=4 V=4", CacheLayer_quant, {"k_bits": 4, "v_bits": 4}),
    ]

    for seq_len in [512, 1024, 2048]:
        print(f"\n  Context: {seq_len}")
        for label, lt, kw in configs:
            ppl, count = run_with_cache(model, tokenizer, eval_tokens, label, lt, kw,
                                        seq_len=seq_len, max_windows=20)
            print(f"    {label:<20} PPL={ppl:.4f} ({count} tokens)")

    # ── TEST 2: Asymmetric K/V ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 2: Asymmetric K/V bit widths (context=1024)")
    print(f"{'='*70}")

    for k_bits, v_bits in [(4, 4), (4, 3), (4, 2), (3, 3), (3, 2), (2, 2)]:
        ppl, count = run_with_cache(model, tokenizer, eval_tokens, f"K={k_bits} V={v_bits}",
                                    CacheLayer_turboquant, {"k_bits": k_bits, "v_bits": v_bits},
                                    seq_len=1024, max_windows=20)
        print(f"  TQ K={k_bits} V={v_bits}  PPL={ppl:.4f}")

    # ── TEST 3: Generation stress (1000 tokens) ──────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 3: 1000-token generation stress test")
    print(f"{'='*70}")

    for label, lt, kw in [
        ("FP16", CacheLayer_fp16, {}),
        ("TQ K=4 V=4", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 4}),
        ("TQ K=4 V=3", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 3}),
    ]:
        cache = make_cache(model, 2048 + 256, lt, **kw)  # +256 for prompt headroom
        text, speed, unique_last = generate_tokens(model, cache, tokenizer, 1000)
        print(f"\n  {label}:")
        print(f"    Speed: {speed:.1f} tok/s")
        print(f"    Unique in last 100: {unique_last}/100")
        print(f"    Last 200 chars: ...{text[-200:]}")
        cache.detach_from_model()
        del cache
        torch.cuda.empty_cache()

    # Done
    model.unload()
    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
