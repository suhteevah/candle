//! Tiled cross-entropy for LLM training.
//!
//! # Why this exists
//!
//! Forward through a 1.5B model produces logits of shape `[B, L, V]`. At
//! V=151936 (Qwen2.5 vocab), B=1, L=256, fp32: 151MB per micro-batch JUST
//! for the logits, sitting there for the entire backward pass. For L=512,
//! it's 300MB. This single allocation is often the largest single VRAM
//! consumer during LoRA training — larger than any layer's activations.
//!
//! # What we do
//!
//! Split the sequence dimension into chunks. For each chunk:
//!   1. Compute logits on just that chunk via `lm_head.forward(xs_chunk)`.
//!   2. Compute masked cross-entropy for that chunk.
//!   3. Weight-sum into a running total.
//!
//! The chunk logits + intermediate log-softmax tensors drop at the end of
//! each iteration (Rust RAII), so peak memory = `chunk × V` instead of
//! `L × V`. With chunk=32, L=256, we reduce peak logits footprint 8×.
//!
//! # Correctness argument
//!
//! Cross-entropy is a sum over token positions of per-token NLL values,
//! normalized by the count of positions with mask=1. Splitting the sum
//! into chunks and accumulating is arithmetically identical — modulo
//! floating-point associativity, which at fp32 accumulation precision is
//! well within training noise.
//!
//! # Gradient flow
//!
//! We compute loss on each chunk via candle's standard differentiable ops.
//! Each chunk's loss is a node in the autograd graph, back-linked to the
//! chunk's logits, which link to `xs_chunk`, which links to the full
//! pre-chunking hidden state (via narrow). The final summed+scaled loss
//! has backward lineage to ALL chunks' hidden state sources, so gradient
//! flows correctly to every LoRA adapter upstream.

use candle::{DType, Module, Result, Tensor, D};
use candle_nn::ops;

/// Tile-aware cross-entropy. `hidden` is `[B, L, H]` (post-norm pre-
/// lm_head); `lm_head` is the final projection module; `targets` and
/// `loss_mask` are `[B, L]`. Returns a scalar loss averaged over unmasked
/// positions.
///
/// The chunking is applied on the sequence dimension. `chunk_size <= 0`
/// disables chunking and falls back to a single-shot compute (equivalent to
/// the non-tiled masked_cross_entropy).
pub fn tiled_cross_entropy(
    hidden: &Tensor,
    lm_head: &candle_nn::Linear,
    targets: &Tensor,
    loss_mask: &Tensor,
    chunk_size: usize,
) -> Result<Tensor> {
    let (b_sz, seq_len, _h) = hidden.dims3()?;
    let (tb, tl) = targets.dims2()?;
    assert_eq!(
        (b_sz, seq_len),
        (tb, tl),
        "hidden and targets must agree on B and L"
    );

    // Single-shot path — useful for short sequences or debugging.
    if chunk_size == 0 || chunk_size >= seq_len {
        let logits = hidden.apply(lm_head)?;
        return compute_masked_ce(&logits, targets, loss_mask);
    }

    // Keep EVERYTHING device-side. Previous version forced 2+ GPU→CPU syncs
    // per chunk via .to_scalar() — at chunk=32, L=256 that's 16+ syncs per
    // micro-batch, obliterating pipelining (~10% GPU util). Lesson: never
    // .to_scalar() inside a training hot path. The denom division happens
    // once, as a tensor op, at the end.
    let mask_f = loss_mask.to_dtype(DType::F32)?;
    let denom_sum = mask_f.sum_all()?; // device-side scalar tensor
    // Add a floor of 1.0 to avoid div-by-zero without syncing. Use
    // maximum(denom_sum, ones_like) — fully on device.
    let one_scalar = Tensor::new(1.0f32, hidden.device())?;
    let denom_safe = denom_sum.broadcast_maximum(&one_scalar)?;

    let mut nll_sum: Option<Tensor> = None;
    let mut start = 0usize;
    while start < seq_len {
        let this = (seq_len - start).min(chunk_size);
        let h_chunk = hidden.narrow(1, start, this)?;
        let t_chunk = targets.narrow(1, start, this)?;
        let m_chunk = loss_mask.narrow(1, start, this)?;

        // We USED to skip all-zero-mask chunks here, but the check required
        // a sync. Instead, always compute — per-chunk cost is small
        // compared to the sync we were saving. For a typical matt-voice
        // pair most chunks have at least some mask=1 positions anyway.
        let logits_chunk = h_chunk.apply(lm_head)?;
        let chunk_nll_sum = compute_masked_ce_sum(&logits_chunk, &t_chunk, &m_chunk)?;
        nll_sum = Some(match nll_sum {
            Some(s) => (s + chunk_nll_sum)?,
            None => chunk_nll_sum,
        });
        start += this;
    }

    let total = nll_sum.unwrap_or(denom_sum.zeros_like()?);
    // Device-side division. Result is a scalar tensor with full autograd
    // lineage through every chunk's logits → lm_head input → backward OK.
    total.broadcast_div(&denom_safe)
}

/// Non-tiled masked cross-entropy — matches the ORIGINAL masked_cross_entropy
/// from main.rs exactly (one .to_scalar for denom per call, compensated by
/// not having any other per-chunk sync). This path is well-exercised and
/// fast; don't reinvent the wheel for the single-shot case.
fn compute_masked_ce(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let logits = if logits.dtype() == DType::F32 {
        logits.clone()
    } else {
        logits.to_dtype(DType::F32)?
    };
    let (_b, _l, vocab) = logits.dims3()?;
    let logits_2d = logits.reshape(((), vocab))?;
    let targets_1d = targets.reshape(((),))?;
    let mask_1d = mask.reshape(((),))?.to_dtype(DType::F32)?;
    let log_probs = ops::log_softmax(&logits_2d, 1)?;
    let picked = log_probs.gather(&targets_1d.unsqueeze(1)?, 1)?.squeeze(1)?;
    let num = (picked * &mask_1d)?.sum_all()?;
    let denom = mask_1d.sum_all()?;
    let denom_f = denom.to_scalar::<f32>()?.max(1.0) as f64;
    (num.affine(-1.0, 0.0)? / denom_f).map_err(Into::into)
}

/// Returns the UNNORMALIZED sum of per-token NLL over mask-positive spots
/// in a chunk. The outer tiled CE divides by the cross-chunk mask total
/// once at the end.
fn compute_masked_ce_sum(
    logits_chunk: &Tensor,
    targets_chunk: &Tensor,
    mask_chunk: &Tensor,
) -> Result<Tensor> {
    let logits = if logits_chunk.dtype() == DType::F32 {
        logits_chunk.clone()
    } else {
        logits_chunk.to_dtype(DType::F32)?
    };
    let (_b, _l, vocab) = logits.dims3()?;
    let logits_2d = logits.reshape(((), vocab))?;
    let targets_1d = targets_chunk.reshape(((),))?;
    let mask_1d = mask_chunk.reshape(((),))?.to_dtype(DType::F32)?;
    let log_probs = ops::log_softmax(&logits_2d, 1)?;
    let picked = log_probs.gather(&targets_1d.unsqueeze(1)?, 1)?.squeeze(1)?;
    // -picked sums the NLL contribution; multiply by mask first to zero
    // non-loss positions.
    let weighted = (picked * &mask_1d)?.sum_all()?;
    weighted.affine(-1.0, 0.0)
}

// silence dead-import warning for D until we add a fused-backward variant
#[allow(dead_code)]
fn _type_fence(_d: D) {}
