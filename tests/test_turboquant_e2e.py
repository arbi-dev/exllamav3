"""
End-to-end test: run Qwen3-0.6B-exl3 with TurboQuant KV cache.
Compares FP16 cache vs TurboQuant cache outputs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time

from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_fp16, CacheLayer_turboquant

MODEL_DIR = "/mnt/k8scache/models/Qwen3.5-35B-A3B-exl3-4.09bpw"

print("=" * 60)
print("TurboQuant End-to-End Test")
print("=" * 60)

gpu_name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"GPU: {gpu_name} (SM{cc[0]}{cc[1]})")
print(f"Model: {MODEL_DIR}")

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)

cl = model.get_cache_layers()
attn0 = cl[0]
print(f"\nModel: {len(cl)} layers, {attn0.num_kv_heads} KV heads, head_dim={attn0.head_dim}")

tokenizer = Tokenizer.from_config(config)
prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt)
vocab = tokenizer.get_id_to_piece_list()

def generate(cache, label, num_tokens=50):
    """Run prefill + generate num_tokens with greedy sampling."""
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, 2048),
    }
    model.prefill(input_ids=input_ids[:, :-1], params=params)

    generated = []
    context = input_ids.clone()

    t0 = time.time()
    for i in range(num_tokens):
        params = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": context.shape[-1] - 1,
            "batch_shape": (1, 2048),
            "recurrent_states": params.get("recurrent_states"),
        }
        logits = model.forward(input_ids=context[:, -1:], params=params)
        token_id = logits[:, -1, :].argmax(dim=-1).item()
        generated.append(token_id)
        context = torch.cat((context, torch.tensor([[token_id]])), dim=-1)
    t1 = time.time()

    text = "".join(vocab[t] for t in generated)
    speed = num_tokens / (t1 - t0)
    print(f"\n--- {label} ---")
    print(f"Output: {text}")
    print(f"Speed: {speed:.1f} tok/s")
    return generated


# ── Test 1: FP16 baseline ────────────────────────────────────────────────────
cache_fp16 = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_fp16)
model.load(progressbar=True)

generated_fp16 = generate(cache_fp16, "FP16 Cache (Baseline)")

cache_fp16.detach_from_model()
model.unload()
del cache_fp16
torch.cuda.empty_cache()

# ── Test 2: TurboQuant 4-bit K, 4-bit V ──────────────────────────────────────
cache_tq = Cache(model, max_num_tokens=2048, layer_type=CacheLayer_turboquant, k_bits=4, v_bits=4)
model.load(progressbar=True)

fp16_bytes = 2 * len(cl) * 2048 * attn0.num_kv_heads * attn0.head_dim * 2
tq_layer = cache_tq.layers[cl[0].layer_idx]
tq_bytes = len(cl) * 2048 * attn0.num_kv_heads * (tq_layer.k_packed_dim + tq_layer.v_packed_dim + 8)  # +8 for norms
print(f"\nMemory: FP16={fp16_bytes/1024/1024:.1f}MB, TQ={tq_bytes/1024/1024:.1f}MB, {fp16_bytes/tq_bytes:.1f}x compression")

generated_tq = generate(cache_tq, "TurboQuant K=4bit V=4bit")

# ── Compare ──────────────────────────────────────────────────────────────────
print("\n--- Comparison ---")
match_count = sum(1 for a, b in zip(generated_fp16, generated_tq) if a == b)
total = min(len(generated_fp16), len(generated_tq))
print(f"Token match: {match_count}/{total} ({match_count/total*100:.0f}%)")

if generated_fp16 == generated_tq:
    print("EXACT MATCH!")
else:
    for i, (a, b) in enumerate(zip(generated_fp16, generated_tq)):
        if a != b:
            print(f"First divergence at token {i}: FP16='{vocab[a]}' vs TQ='{vocab[b]}'")
            break

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
