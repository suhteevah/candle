//! Gradient checkpointing (activation recomputation) for qwen-lora-train.
//!
//! # Why this module exists
//!
//! At full activation retention, Qwen2.5-1.5B LoRA training on an 8GB GPU
//! caps at seq=128 with q/k/v/o target modules. Maxwell 4GB cards can't fit
//! training at all. The activation footprint is dominated by the 28
//! transformer layers — each keeps ~5-8 intermediate tensors alive for
//! backward, most of them shape `[B, L, hidden]` or `[B, n_heads, L, L]`.
//!
//! This module implements **layer-boundary activation recomputation**:
//! during forward, we keep only the per-layer input tensors and detach each
//! layer's output; during backward, we re-run each layer's forward with
//! autograd and drive its local backward with the upstream gradient
//! accumulated from the next layer. The pattern is the same as PyTorch's
//! `torch.utils.checkpoint.checkpoint_sequential` — applied at the
//! `DecoderLayer` granularity.
//!
//! # Why this isn't a CustomOp1
//!
//! Candle's `CustomOp1::bwd` signature only receives `(arg, res, grad_res)`
//! and returns a single gradient tensor. There's no access to the outer
//! backward's `GradStore`, so the inner sub-forward's Var gradients have
//! nowhere to accumulate. The enabling primitive was
//! `Tensor::backward_into(&mut GradStore, Option<Tensor>)` (added to
//! candle-core in an earlier commit in this branch) — that gives us
//! composable backward passes. We drive the checkpointing orchestration
//! from the training loop, not from inside the model's forward.
//!
//! # API shape (planned)
//!
//! ```ignore
//! let ctx = CheckpointContext::new(num_layers);
//! // during forward:
//! for layer in &mut model.layers {
//!     let layer_input = ctx.record_input(current.detach()?)?;
//!     let layer_output = layer.forward(&layer_input, attn_mask, 0, training)?;
//!     current = ctx.record_output(layer_output)?;  // detaches internally
//! }
//! // ... norm + lm_head + loss as usual ...
//!
//! // during backward:
//! let mut grads = GradStore::new();
//! loss.backward_into(&mut grads, None)?;
//! // walk layers in reverse, recompute and accumulate
//! ctx.backward_through_layers(&mut grads, |i, saved_input, upstream| {
//!     let fresh_out = model.layers[i].forward(saved_input, attn_mask, 0, true)?;
//!     fresh_out.backward_into(grads, Some(upstream))?;
//!     Ok(())
//! })?;
//! ```
//!
//! # Memory argument
//!
//! - Without checkpointing: 28 × layer_activations kept in memory for the
//!   whole forward + backward = ~3-4GB at seq=128 for Qwen2.5-1.5B.
//! - With checkpointing: only 28 × layer_input tensors kept (shape
//!   `[B, L, hidden]` each, bf16 = 2B × 128 × 1536 × 2 ≈ 500KB per layer,
//!   14MB total) plus one layer's full activations alive at a time during
//!   backward recompute (~150MB). Peak memory reduction: 10-20×.
//!
//! # Compute cost
//!
//! Forward runs 2× per layer (once normally, once during backward recompute).
//! For a fully recomputed model that's ~33% wall-clock overhead on training.
//! Acceptable in exchange for fitting on Maxwell.
//!
//! # Status
//!
//! This file is the scaffolded API + memory-accounting design. The actual
//! implementation is the next commit. Deliberately separated from the main
//! file so the API can be reviewed before the model integration lands.

use anyhow::Result;
use candle::Tensor;
use candle_nn::VarMap;

/// Per-layer state captured during the forward pass, consumed during the
/// reverse-order backward recompute.
#[derive(Debug)]
pub struct LayerCheckpoint {
    /// Detached copy of the layer's input. This is what the recompute
    /// forward will consume. Holds a tensor at detach-leaf status so the
    /// graph before this layer is already torn down.
    pub saved_input: Tensor,
    /// The layer's original forward output (still in its Op graph) — used
    /// only for TensorId lookups in the outer backward's GradStore. We don't
    /// hold onto the op graph for memory reasons; only the TensorId is
    /// load-bearing.
    pub output_id: candle::TensorId,
}

/// Per-training-step orchestrator. Call `record_boundary` after each
/// checkpointed unit of the forward (e.g., after each DecoderLayer); then
/// during backward call `backward_through_segments` in reverse to drive
/// the recompute.
#[derive(Debug, Default)]
pub struct CheckpointContext {
    pub checkpoints: Vec<LayerCheckpoint>,
}

impl CheckpointContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a checkpointable unit has just produced `output` from
    /// `input`. Returns a *detached* clone of `output` suitable for use as
    /// input to the next unit. The original `output` is kept briefly (for
    /// its id) and then drops — its intermediate activations die with it.
    pub fn record_boundary(&mut self, input: Tensor, output: Tensor) -> Result<Tensor> {
        let saved_input = input.detach();
        let output_id = output.id();
        let detached_output = output.detach();
        self.checkpoints.push(LayerCheckpoint {
            saved_input,
            output_id,
        });
        Ok(detached_output)
    }

    /// Walk checkpoints in reverse, calling `recompute` for each. `recompute`
    /// is expected to:
    ///   1. Re-run the unit's forward on `saved_input` with autograd tracking.
    ///   2. Call the fresh output's `backward_into(grads, Some(upstream))`.
    /// We handle the upstream-grad plumbing between segments.
    pub fn backward_through_segments<F>(
        &self,
        grads: &mut candle::backprop::GradStore,
        mut recompute: F,
    ) -> Result<()>
    where
        F: FnMut(
            usize,
            &Tensor,
            Tensor,
            &mut candle::backprop::GradStore,
        ) -> Result<()>,
    {
        let _ = grads;
        let _ = &mut recompute;
        // TODO: implement in the next commit. Pseudocode:
        //
        //   for i in (0..self.checkpoints.len()).rev() {
        //       let cp = &self.checkpoints[i];
        //       let upstream = grads
        //           .remove_by_id(cp.output_id)
        //           .ok_or_else(|| anyhow::anyhow!("no grad at layer {i} output — graph was cut somewhere unexpected"))?;
        //       recompute(i, &cp.saved_input, upstream, grads)?;
        //       // after recompute, grads now has an entry keyed on
        //       // cp.saved_input.id() that we'll forward as the NEXT
        //       // layer's upstream grad by pulling grads.remove_by_id(
        //       // self.checkpoints[i-1].output_id).
        //       //
        //       // But wait — saved_input of layer i is NOT the same tensor
        //       // as output of layer i-1. They share values but are different
        //       // TensorIds because of the detach. We need to copy/alias the
        //       // gradient: whatever arrived at saved_input[i], use as
        //       // upstream for output[i-1].
        //       //
        //       // This is why `record_boundary` keeps output_id explicitly —
        //       // so we can move the grad from saved_input[i].id to
        //       // checkpoints[i-1].output_id between segments.
        //   }
        anyhow::bail!(
            "CheckpointContext::backward_through_segments is a scaffold — \
             implementation lands in the next commit. See module docstring."
        )
    }
}

/// Sanity-check: the checkpointing design relies on `grads.remove_by_id`
/// (or equivalent) working cleanly across `backward_into` invocations. If
/// candle-core doesn't expose an id-keyed lookup we need to add one — this
/// is the only remaining core API tweak for gradient checkpointing to work.
#[allow(dead_code)]
fn _api_gap_note(_: &VarMap) {
    // See backprop.rs: GradStore already has get_id(TensorId) and
    // insert_id(TensorId, Tensor). remove_by_id does NOT exist as of the
    // refactor in 5eba650 — we'll add it alongside the orchestration impl.
}
