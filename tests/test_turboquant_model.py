"""
TurboQuant integration test with a real model.

Usage:
    python tests/test_turboquant_model.py -m /path/to/exl3/model [-l 512] [-g 50]

Tests:
1. Greedy generation: compare FP16 vs TurboQuant token-by-token
2. WikiText-2 perplexity: FP16 vs TurboQuant vs exl3 native (if datasets available)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import time

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant, CacheLayer_turboquant


@torch.inference_mode()
def generation_test(model, tokenizer, layer_type, cache_kwargs, label, num_tokens=50, max_cache=2048):
    vocab = tokenizer.get_id_to_piece_list()
    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt)

    cache = Cache(model, max_num_tokens=max_cache, layer_type=layer_type, **cache_kwargs)
    device = torch.device("cuda:0")
    for layer in cache.layers.values():
        if layer.device is None:
            layer.alloc(device)

    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, max_cache)}
    model.prefill(input_ids=input_ids[:, :-1], params=params)

    generated = []
    context = input_ids.clone()
    t0 = time.time()
    for _ in range(num_tokens):
        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": context.shape[-1] - 1,
                  "batch_shape": (1, max_cache), "recurrent_states": params.get("recurrent_states")}
        logits = model.forward(input_ids=context[:, -1:], params=params)
        token_id = logits[:, -1, :].argmax(dim=-1).item()
        generated.append(token_id)
        context = torch.cat((context, torch.tensor([[token_id]])), dim=-1)
    elapsed = time.time() - t0

    text = "".join(vocab[t] for t in generated)
    cache.detach_from_model()
    del cache
    torch.cuda.empty_cache()
    return generated, text, num_tokens / elapsed


@torch.inference_mode()
def ppl_test(model, tokenizer, eval_tokens, layer_type, cache_kwargs, seq_len, max_windows=30):
    import torch.nn.functional as F
    import math

    vocab_size = tokenizer.actual_vocab_size
    stride = seq_len // 2
    logprob_sum = 0.0
    logprob_count = 0

    cache = Cache(model, max_num_tokens=seq_len, layer_type=layer_type, **cache_kwargs)
    device = torch.device("cuda:0")
    for layer in cache.layers.values():
        if layer.device is None:
            layer.alloc(device)

    for i, start in enumerate(range(0, eval_tokens.shape[1] - seq_len, stride)):
        if i >= max_windows:
            break
        ids = eval_tokens[:, start:start + seq_len]
        params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0, "batch_shape": (1, seq_len)}
        logits = model.forward(ids, params)
        eval_start = max(0, seq_len - stride)
        logits = logits[:, eval_start:-1, :vocab_size].float() + 1e-10
        lp = F.log_softmax(logits, dim=-1)
        tgt = ids[:, eval_start + 1:].to(lp.device)
        logprob_sum += lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
        logprob_count += tgt.numel()
        del logits, lp, tgt

    cache.detach_from_model()
    del cache
    torch.cuda.empty_cache()
    return math.exp(-logprob_sum / logprob_count), logprob_count


def main():
    parser = argparse.ArgumentParser(description="TurboQuant integration test")
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-g", "--gen_tokens", type=int, default=50)
    parser.add_argument("-l", "--seq_len", type=int, default=512)
    args = parser.parse_args()

    print("=" * 60)
    print("TurboQuant Integration Test")
    print("=" * 60)

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cl = model.get_cache_layers()
    print(f"Model: {args.model_dir}")
    print(f"  {len(cl)} attn layers, {cl[0].num_kv_heads} KV heads, head_dim={cl[0].head_dim}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load model once with a dummy cache
    dummy = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_fp16)
    model.load(progressbar=True)
    dummy.detach_from_model()
    del dummy
    torch.cuda.empty_cache()

    # ── Generation test ───────────────────────────────────────────────────────
    print(f"\n--- Generation ({args.gen_tokens} tokens, greedy) ---")

    configs = [
        ("FP16", CacheLayer_fp16, {}),
        ("TQ K=4 V=4", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 4}),
        ("TQ K=4 V=3", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 3}),
        ("exl3 K=4 V=4", CacheLayer_quant, {"k_bits": 4, "v_bits": 4}),
    ]

    results = {}
    for label, lt, kw in configs:
        gen, text, speed = generation_test(model, tokenizer, lt, kw, label, args.gen_tokens)
        results[label] = gen
        print(f"\n  {label} ({speed:.1f} tok/s):")
        print(f"    {text[:120]}...")

    fp16_gen = results["FP16"]
    for label, gen in results.items():
        match = sum(1 for a, b in zip(fp16_gen, gen) if a == b)
        print(f"  {label}: {match}/{args.gen_tokens} token match vs FP16")

    # ── PPL test (if datasets available) ──────────────────────────────────────
    try:
        from datasets import load_dataset
        print(f"\n--- Perplexity (WikiText-2, context={args.seq_len}) ---")
        dataset_text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
        eval_tokens = tokenizer.encode(dataset_text)[:, :32768]

        for label, lt, kw in configs:
            ppl, count = ppl_test(model, tokenizer, eval_tokens, lt, kw, args.seq_len)
            print(f"  {label:<20} PPL={ppl:.4f} ({count} tokens)")
    except ImportError:
        print("\n  (Skipping PPL test — install 'datasets' package)")

    model.unload()
    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    main()
