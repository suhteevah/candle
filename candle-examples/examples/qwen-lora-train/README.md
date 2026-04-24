# qwen-lora-train

**Status:** scaffold / in-progress. Not yet upstream-ready.

A LoRA fine-tuning example for Qwen2.5-class models. Targets **older GPU architectures** that unsloth / bitsandbytes don't support: **Maxwell (CC 5.0+), Pascal (CC 6.0+), Turing (CC 7.5+), Ampere (CC 8.0+)**. Also runs on CPU for validation.

Loads a GGUF-quantized base model (Q4_K_M, Q5_0, etc.) + attaches fp16/fp32 LoRA adapters to attention and MLP projections. Only adapter weights are trained; base is frozen. Exports HuggingFace-PEFT-compatible adapter checkpoints (`adapter_config.json` + `adapter_model.safetensors`) that load cleanly in any peft-aware inference stack.

## Why this exists

- unsloth requires Ampere+ (bf16 WMMA, FlashAttention-2). Millions of deployed GPUs are older than that.
- bitsandbytes dropped Maxwell in 2023.
- candle itself works fine on CC 5.0+ (soft feature gates, not hard build gates), but had no training example beyond MNIST.
- Gap: a LoRA trainer that actually runs on a 4GB GTX 980.

## Planned architecture

```
main.rs          CLI, training-loop orchestration, device + dtype selection
qwen2_lora.rs    Forked quantized_qwen2 with LoRA taps on q/k/v/o + MLP
lora.rs          Reusable LoRA Linear module (rank-r low-rank adapter)
dataset.rs       JSONL dataset loader: {"context":...,"matt":...} pairs
adapter.rs       PEFT-compatible safetensors export + adapter_config.json
```

## Planned hardware targets

| GPU | CC | Status | Notes |
|---|---|---|---|
| RTX 3070 Ti | 8.6 | Dev platform, build verified 2026-04-23 | Ampere — easiest |
| Tesla P100 | 6.0 | Target (cnc, pending) | Pascal — no bf16, fp16 only |
| GTX 980 | 5.2 | Stretch target | 4GB VRAM, needs Q4 base + rank-8 adapter + grad checkpointing |

## Planned CLI

```
cargo run --release --example qwen-lora-train --features cuda -- \
    --base-model /path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --tokenizer /path/to/tokenizer.json \
    --dataset /path/to/matt-voice.jsonl \
    --output-dir ./adapters/matt-voice-v1 \
    --rank 16 \
    --alpha 32 \
    --target-modules q_proj,k_proj,v_proj,o_proj \
    --batch-size 1 \
    --grad-accum-steps 16 \
    --learning-rate 2e-4 \
    --max-steps 500
```

## What's in the scaffold right now

- [x] CLI + arg parsing
- [x] Dataset loader for matt-voice JSONL
- [x] `LoRALinear` module (rank-r, xavier init for A, zero init for B)
- [x] PEFT-compatible adapter export (scaffold)
- [ ] Base model with LoRA taps (`qwen2_lora.rs`) — **TODO: fork `quantized_qwen2.rs` and inject taps**
- [ ] Training loop with AdamW + gradient accumulation
- [ ] Learning-rate warmup + cosine schedule
- [ ] Checkpoint resume
- [ ] Gradient checkpointing (needed for Maxwell 4GB)
- [ ] QLoRA-style: quantized base + fp16 adapter (candle already supports quantized read via `gguf_file.rs` — need to validate backward pass doesn't require dequant)

## Why fork `quantized_qwen2.rs`

Candle's `ModelWeights` in `candle_transformers::models::quantized_qwen2` is inference-only — it doesn't expose attention projection outputs in a way that lets us inject a LoRA delta. Cleanest prototype: local copy + splice LoRA into `forward_attn` at the q/k/v/o projections. Long-term candle could expose a hook API, but that's a bigger design conversation than this example needs.

## Related

- matt-voice training pipeline: `J:\matt-voice\` (Discord corpus collection, current Python/unsloth trainer)
- matt-voice listener (rust): `J:\matt-voice\listener\`
- Candle reference: `candle-examples/examples/quantized-qwen2-instruct` (inference)
- Candle reference: `candle-examples/examples/mnist-training` (training idioms)
