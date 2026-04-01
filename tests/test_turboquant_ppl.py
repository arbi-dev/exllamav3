"""
Perplexity comparison: FP16 cache vs TurboQuant vs exl3 native quantized cache.
Uses WikiText-2 test set, cached forward passes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant, CacheLayer_turboquant


def get_wikitext2_tokens(tokenizer, max_tokens=None):
    """Load and tokenize WikiText-2 test set."""
    print("Loading WikiText-2 test set...")
    dataset_text = "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    tokens = tokenizer.encode(dataset_text)
    if max_tokens and max_tokens > 0:
        tokens = tokens[:, :max_tokens]
    print(f"  {tokens.shape[1]} tokens")
    return tokens


@torch.inference_mode()
def eval_perplexity(model, cache, tokenizer, eval_tokens, seq_len=512, stride=256):
    """Evaluate perplexity using cached forward passes with sliding window."""
    vocab_size = tokenizer.actual_vocab_size
    num_tokens = eval_tokens.shape[1]

    logprob_sum = 0.0
    logprob_count = 0
    num_windows = 0

    for start in range(0, num_tokens - seq_len, stride):
        end = start + seq_len
        input_ids = eval_tokens[:, start:end]

        # Prefill entire sequence
        params = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": 0,
            "batch_shape": (1, seq_len),
        }
        logits = model.forward(input_ids, params)

        # Compute log probs for the stride portion (to avoid double-counting)
        eval_start = max(0, seq_len - stride)
        logits = logits[:, eval_start:-1, :vocab_size].float()
        logits += 1e-10
        log_probs = F.log_softmax(logits, dim=-1)
        target_ids = input_ids[:, eval_start + 1:].to(log_probs.device)
        target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        logprob_sum += target_log_probs.sum().item()
        logprob_count += target_ids.numel()
        num_windows += 1

        if num_windows % 10 == 0:
            ppl_so_far = math.exp(-logprob_sum / logprob_count)
            print(f"  window {num_windows}: ppl={ppl_so_far:.4f} ({logprob_count} tokens)")

        del logits, log_probs, target_log_probs, target_ids
        torch.cuda.empty_cache()

        if num_windows >= 50:  # Limit for speed
            break

    mean_log_prob = logprob_sum / logprob_count
    perplexity = math.exp(-mean_log_prob)
    return perplexity, logprob_count


def run_eval(model_dir, max_tokens=32768, seq_len=512):
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)

    cl = model.get_cache_layers()
    attn0 = cl[0]
    print(f"\nModel: {model_dir}")
    print(f"  {len(cl)} attn layers, {attn0.num_kv_heads} KV heads, head_dim={attn0.head_dim}")

    eval_tokens = get_wikitext2_tokens(tokenizer, max_tokens)

    configs = [
        ("FP16", CacheLayer_fp16, {}),
        ("TurboQuant K=4 V=4", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 4}),
        ("TurboQuant K=4 V=3", CacheLayer_turboquant, {"k_bits": 4, "v_bits": 3}),
        ("TurboQuant K=3 V=3", CacheLayer_turboquant, {"k_bits": 3, "v_bits": 3}),
        ("exl3 native K=4 V=4", CacheLayer_quant, {"k_bits": 4, "v_bits": 4}),
        ("exl3 native K=3 V=3", CacheLayer_quant, {"k_bits": 3, "v_bits": 3}),
    ]

    results = []
    for label, layer_type, kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        cache = Cache(model, max_num_tokens=seq_len, layer_type=layer_type, **kwargs)
        model.load()

        t0 = time.time()
        ppl, count = eval_perplexity(model, cache, tokenizer, eval_tokens, seq_len=seq_len)
        t1 = time.time()

        results.append((label, ppl, count, t1 - t0))
        print(f"  Perplexity: {ppl:.4f} ({count} tokens, {t1-t0:.1f}s)")

        cache.detach_from_model()
        model.unload()
        del cache
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'PPL':>10} {'Tokens':>10} {'Time':>8}")
    print("-" * 60)
    for label, ppl, count, elapsed in results:
        print(f"{label:<30} {ppl:>10.4f} {count:>10} {elapsed:>7.1f}s")

    # Delta from FP16
    if results:
        fp16_ppl = results[0][1]
        print(f"\n{'Method':<30} {'PPL delta':>10}")
        print("-" * 42)
        for label, ppl, _, _ in results:
            delta = ppl - fp16_ppl
            print(f"{label:<30} {delta:>+10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, required=True)
    parser.add_argument("-t", "--max_tokens", type=int, default=32768)
    parser.add_argument("-l", "--seq_len", type=int, default=512)
    args = parser.parse_args()
    run_eval(args.model_dir, args.max_tokens, args.seq_len)
