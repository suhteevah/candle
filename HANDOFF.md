# Candle Fork Handoff

## Last Updated
2026-04-24

## Project Status
🟢 **7B QLoRA training on 8GB working end-to-end with real loss decline on matt-voice corpus.**

## What Was Done This Session

Single multi-hour push from "candle scaffold + plan" to "verified 7B fine-tuning of matt-voice on 8GB consumer GPU."

### candle-core changes
- `Tensor::backward_into(&mut GradStore, Option<Tensor>)` — composable backward primitive (commit `5eba650`)
- `GradStore::remove_by_id(TensorId)` for the gradient checkpointing flow
- **`QMatMul` backward** via `QMatmulBwdOp` wrapper + analytical `QTensor::bwd` (commit `73b6c78`) — **the QLoRA unlock**. Reshape-safe for 3D grad_res via 2D fold/unfold.

### qwen-lora-train (the example)
| Module | Purpose |
|---|---|
| `main.rs` | CLI + TrainModel enum dispatch on `--gguf` / `--base-dir` |
| `qwen2_lora.rs` | fp16 safetensors path with LoRA + checkpointing |
| `qwen2_lora_quantized.rs` | **NEW**: GGUF path with QMatMul+LoRA, lm_head pre-dequant to f16 |
| `checkpoint.rs` | Last-layer-not-detached gradient checkpointing |
| `fused_ops.rs` | **NEW**: fused softmax + fused RMSNorm with analytical bwd, detached y cache |
| `lora.rs` | Reusable LoRA adapter module |
| `bench.rs` | `--benchmark` mode: 3 warmup + 20 measured steps, single-line BENCH json output |
| `dataset.rs` | matt-voice JSONL loader |
| `adapter.rs` | PEFT-compatible safetensors exporter |
| `xentropy.rs` / `prefetch.rs` | scaffolded but DISABLED (caused stalls; see Notes) |

### Verified results

**Benchmark progression on Qwen2.5-7B Q4_K_M, rank 8 q/v, GC, 3070 Ti 8GB:**

| Config | tok/s | step ms | peak MB |
|---|---|---|---|
| seq=64 (initial 7B fit) | 5.2 | 47141 | 7947 |
| + fused softmax | 5.3 | 46859 | 7914 |
| + fused RMSNorm (detached) | 5.4 | 46171 | 7913 |
| **seq=128** | 5.5 | 53933 | 7919 |
| **seq=256** | 5.7 | 53111 | 7918 |
| **seq=512** | 5.8 | 52180 | 7918 |

Memory is essentially constant across seq lengths — QMatMul dequant transients dominate.

**Real training run** (q/k/v/o, seq=128, 10 steps): loss **9.23 → 3.21 in 7 steps**, all 112 B matrices populated, PEFT-compatible adapter saved (19.25 MB).

## Current State

### Working
- fp16 (safetensors) training path: validated on Qwen2.5-1.5B
- **QLoRA (GGUF) training path: validated on Qwen2.5-3B and Qwen2.5-7B**
- Gradient checkpointing
- State checkpointing (`--save-every` / `--resume-from`)
- `--benchmark` mode for reproducible measurement
- TrainModel enum dispatch on `--gguf`
- Fused softmax + fused RMSNorm (with analytical backward)
- Single-binary deployment for inference (`Model::forward()` already supports kv-cached generation)

### Stubbed / Disabled
- `--prefetch-queue > 0` — causes ~8% GPU util stall (disabled by default, defaults to 0)
- `--ce-chunk-size > 0` — same stall (disabled, defaults to 0)
- Multi-shard GGUF — candle's `gguf_file::Content::read` is single-file only; download single-file from bartowski

## Blocking Issues
None for matt-voice training. We can run a real fine-tune NOW.

## What's Next

Prioritized for next session:

1. **Stage 3: Fused RoPE** — same pattern as fused_softmax / fused_rms_norm. RoPE is orthogonal so backward is the same op with negated sin. Expected: small further speedup at this seq length (RoPE is not the bottleneck), but useful for upstream PR completeness.
2. **Flash Attention v2 wiring** — 3070 Ti is Ampere CC 8.6, supported. May be marginal at seq 64-128 but big win at 512+.
3. **Real matt-voice fine-tune** — kick off a 1000+ step run with `--save-every 200` so we get intermediate adapters. Estimate: ~15-20 hours wall-clock for a meaningful pass over 46k pairs.
4. **Inference server** — single-binary candle inference for serving the trained adapter. `Model::forward()` already does the heavy lifting; wrap with axum.
5. **Upstream PR prep** — split commits into clean upstream-able units: (a) `Tensor::backward_into`, (b) `QMatMul` backward, (c) the qwen-lora-train example as a whole.
6. **P100 cnc deployment** — when cables land. 16GB VRAM unlocks bigger configs (q/k/v/o/gate/up/down at seq 256+).

## Notes for Next Session

### Gotchas already documented in memory
- `reference_candle_nobwd_ops.md` — rope/softmax/rms_norm fast paths are no_bwd; use *_slow or fused variants for training
- `reference_candle_gradient_checkpointing.md` — last layer must NOT be detached
- `reference_candle_qmatmul_no_backward.md` — superseded by our 73b6c78
- `reference_candle_7b_qlora_working.md` — first 7B fit
- `reference_candle_7b_matt_voice_training.md` — verified loss decline on matt-voice

### Methodology that's been working
- One feature per commit, bench before + after via `--benchmark`
- Cold-start protocol: kill all qwen-lora-train procs, wait 30s, verify VRAM >7GB free
- If a change doesn't improve tok/s OR peak_vram, revert and diagnose

### Known patterns NOT to repeat
- Don't try to land multiple optimizations simultaneously (the early "hit all of it" pass burned hours)
- Don't try device-side loss accumulation across micro-batches (`accum_loss_tensor = Some(a + &loss)`) — even with detach it pessimizes the allocator catastrophically
- Don't trust "fp16 is fine" — Qwen weights are bf16, casting to fp16 OOMs from outlier overflow

### Reproducible 7B training command

```
GGUF='J:\matt-voice\models\qwen2.5-7b-q4km.gguf'
TOKENIZER='C:\Users\Matt\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306\tokenizer.json'

J:/candle-src/target/release/examples/qwen-lora-train.exe \
  --gguf "$GGUF" --tokenizer "$TOKENIZER" \
  --dataset 'J:\matt-voice\training-data\matt-voice.jsonl' \
  --output-dir 'J:\matt-voice\adapters\matt-voice-7b-qlora' \
  --rank 8 --alpha 16 --target-modules q_proj,k_proj,v_proj,o_proj \
  --batch-size 1 --grad-accum-steps 4 \
  --learning-rate 2e-4 --max-steps 1000 --max-seq-len 128 \
  --log-every 10 --save-every 200 --gradient-checkpoint
```

### Build prerequisites (kokonoe)
- Rust toolchain MSVC: `cd J:\candle-src && rustup override set stable-x86_64-pc-windows-msvc`
- VS BuildTools 2022 vcvars64 sourced via `J:\candle-src\build-cuda.bat` wrapper
- `CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=link.exe` (override global lld-link)
- `CUDA_COMPUTE_CAP=86` for Ampere

### Repo state
- Branch: `matt-voice-lora` on github.com/suhteevah/candle (HEAD: latest commit, see `git log --oneline -5`)
- ~25 commits this session, all pushed
- cnc lockstep with kokonoe via `git pull origin matt-voice-lora` (cnc has the fork as origin)
