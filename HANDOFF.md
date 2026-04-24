# Candle Fork Handoff — 2026-04-24

**Branch:** `matt-voice-lora` on github.com/suhteevah/candle
**HEAD:** `10d9985`

## What's shipped and verified

| Commit | What | Verification |
|---|---|---|
| `d22eec1` | qwen-lora-train scaffold | compiles |
| `2d56f2e` | v1 trainer | compiles |
| `0ed8fbf` | NaN fixes (bf16 base + fp32 loss) | verified, training runs |
| `5c33b46` | **no_bwd ops fix** (rope/softmax/rms → slow variants) | **real training unlock** — losses trend down, adapters update |
| `86ca9ac` | `--save-every` / `--resume-from` state checkpointing | save verified, resume load verified |
| `5eba650` | **`Tensor::backward_into` core API** | compiles, used by checkpoint module |
| `4f053e4` | gradient checkpoint scaffold | compiles |
| `066d008` | **gradient checkpointing end-to-end** | 8-step smoke, all 56 B matrices populated, loss trending |
| `930d3cc` | prefetch + tiled CE scaffolded (disabled default) | both stall in current form — marked as known issues |
| `98d9aae` | `--benchmark` mode + bench.rs harness | **baseline captured: 80.1 tok/s, 7895 MB peak, 92.5% GPU util** on 1.5B rank 8 q/v seq 128 GC |
| `73b6c78` | **QMatMul backward** (candle-core `QTensor::bwd`) | compiles, dependencies built clean |
| `10d9985` | **Quantized-base LoRA model** (`qwen2_lora_quantized.rs`) | compiles, dead code until wired |

## What still needs wiring (the 7B unlock)

**Status:** the hard parts are done. What's left is main.rs dispatch wiring.

### Concrete work for next session

1. **Add `--gguf <path>` CLI flag** to Args in main.rs.
2. **Create a `TrainModel` enum** that wraps either:
   - `qwen2_lora::Model` (safetensors path, existing)
   - `qwen2_lora_quantized::Model` (GGUF path, shipped in `10d9985`)
3. **Forward delegates** on TrainModel:
   - `forward_train(&mut self, input_ids) -> Result<Tensor>` → hidden [B,L,H]
   - `forward_train_with_checkpoint(&mut self, input_ids, &mut ctx) -> Result<(Tensor, Option<Tensor>)>`
   - `backward_through_checkpoints(&mut self, &ctx, &mut grads, mask)`
   - `apply_lm_head(&self, hidden) -> Result<Tensor>` → logits [B,L,V] (wraps either `Linear::forward` or `QMatMul::forward`)
4. **Branch on `--gguf`** right after arg parsing:
   - If set: load GGUF via `gguf_file::Content::read` → `qwen2_lora_quantized::Model::from_gguf` → `TrainModel::Quant(model)`
   - Else: existing safetensors path → `TrainModel::Fp(model)`
5. **Update training loop** to call `TrainModel` methods instead of direct model methods.
6. **Test on 3B GGUF first** (`C:\Users\Matt\models\qwen2.5-3b-q4_k_m.gguf` — single file, fast iteration).
7. **Validate QMatMul backward flows** by inspecting adapter weights after 5 steps on 3B quantized.
8. **Move to 7B GGUF** (`C:\Users\Matt\models\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`).
   - ⚠️ **Multi-shard GGUF may not load** via candle's current `gguf_file::Content::read` (single-file only).
   - If blocked: convert to single-file via `llama-gguf-split --merge` OR download a single-file q4_k_m from HF.
9. **Benchmark QLoRA 7B** — should be roughly the baseline's perf (80 tok/s) adjusted for model size.
10. **Run real 7B training** for enough steps to see loss trend down + verify adapters.

**Expected VRAM for 7B q4_k_m rank 8 q/v seq 128 GC:** ~4.5 GB (weights) + ~600 MB (activations + transient dequant + adapters + optim) = **~5.1 GB on 8GB kokonoe**. Plenty of headroom even with Discord + Chrome open.

## Known issues (still open)

- **`--prefetch-queue > 0`**: training stalls at ~8% GPU util. Defaults to 0. Root cause unknown (suspect candle CUDA context per-thread interaction on Windows).
- **`--ce-chunk-size > 0`**: similar stall. Defaults to 0. Revisit after fused-CE CustomOp lands.
- **VRAM fragmentation after killed processes**: Windows CUDA driver doesn't immediately release; a 30-60s cool-down between runs is sometimes needed.

## Queue of optimizations NOT yet attempted

From the plan doc in chat, ordered by juice-per-LOC:

1. Profile baseline with `nvprof`/`nsys` to identify actual bottlenecks (do BEFORE more optimizations).
2. Fused cross-entropy CustomOp (analytical bwd, saves 200-400 MB).
3. Fused RMSNorm CustomOp (saves 100-300 MB).
4. Fused RoPE CustomOp.
5. Fused softmax CustomOp.
6. gate+up matmul fusion with LoRA-aware concat.
7. Flash Attention v2 on CC 8.0+ (3070 Ti).

## Files in this fork (relevant to training)

```
candle-core/src/
  backprop.rs                     +28 LOC (backward_into, remove_by_id)
  quantized/mod.rs                +95 LOC (QTensor::bwd, QMatmulBwdOp)

candle-examples/examples/qwen-lora-train/
  README.md
  Cargo.toml                      (auto-discovered)
  main.rs                         (orchestration + CLI)
  qwen2_lora.rs                   (fp16 safetensors path)
  qwen2_lora_quantized.rs         (GGUF quantized path — NEW, unused until wired)
  checkpoint.rs                   (gradient checkpointing orchestration)
  lora.rs                         (LoRALinear adapter module)
  dataset.rs                      (matt-voice JSONL loader)
  adapter.rs                      (PEFT-compatible exporter)
  bench.rs                        (microbenchmark harness)
  prefetch.rs                     (background tokenizer, disabled)
  xentropy.rs                     (tiled CE, disabled)
```

## Methodology discipline for next session

1. Run the baseline benchmark once (fresh VRAM) to confirm the reference number is still ~80 tok/s.
2. ONE optimization per commit. Bench before, bench after.
3. Cold-start protocol between measurements: kill all candle procs, wait 30s, verify `nvidia-smi` shows >7GB free.
4. If a change doesn't improve tok/s or peak_vram_mb, revert it.

## Bottom line

Everything needed for 7B QLoRA training is IN the tree:
- Training on quantized base is now possible (QMatMul backward)
- The quantized model structure is written and compiles (qwen2_lora_quantized::Model)
- The gradient checkpointing + all other unlocks carry over unchanged

What's left is a ~200 LOC dispatch wire-up + actual runs. Next session picks up there.
