//! Fused ops with analytical backward, specifically for the training hot
//! path. Each replaces a chain of 5-8 composed differentiable ops with a
//! single CustomOp that caches only what the gradient formula actually
//! needs — saving allocator churn, kernel launches, and VRAM.
//!
//! Scoping: these live in the example to prove them out. Once validated
//! they're good candidates to upstream into candle-nn.
//!
//! # Stage 1: Softmax (last dim)
//!
//! Composed reference in candle: `candle_nn::ops::softmax` uses
//! `max_keepdim` + `broadcast_sub` + `exp` + `sum_keepdim` + `broadcast_div`
//! = 5 intermediate tensors kept alive for backward.
//!
//! Fused analytical bwd needs only the softmax output `y`:
//!
//! ```text
//! y_i       = exp(x_i - max) / sum_j exp(x_j - max)
//! dL/dx_i   = y_i * ( dL/dy_i  -  sum_j( dL/dy_j * y_j ) )
//! ```
//!
//! Cache: just `y` (1 tensor). Composed retains 5.
//!
//! Forward impl here re-uses candle's existing `ops::softmax` for
//! convenience (same formula). We pay one composed-ops forward to get `y`,
//! but the backward graph registered is a single CustomOp1 whose bwd
//! runs the analytical formula. Peak memory during backward = just `y`.

use candle::backend::BackendStorage;
use candle::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor, D};

/// Softmax over the last dimension with a compact analytical backward.
///
/// Drop-in replacement for `candle_nn::ops::softmax(xs, D::Minus1)?` on
/// the training hot path. At inference time it's roughly equivalent.
pub fn fused_softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    // For the forward value we can just reuse the existing composed path
    // (no_bwd variant is slightly cheaper but we want graph connectivity
    // for our own bwd callback). Use the standard ops::softmax which
    // returns a tensor y with identity equal to what our analytical bwd
    // expects.
    //
    // Strategy: compute y via candle_nn::ops::softmax (composed), then
    // rewrap via apply_op1 registering SoftmaxBwd which discards the
    // composed graph and uses our analytical bwd instead.
    //
    // This saves memory because the composed intermediates go out of
    // scope after this fn returns — only `y` (the final output of the
    // composed chain) is live, and our CustomOp1 output aliases it.
    let y = candle_nn::ops::softmax(xs, D::Minus1)?;
    // Re-wrap via a custom op whose output is the IDENTITY of y but whose
    // backward uses the analytical formula and depends ONLY on y, not on
    // the composed intermediates.
    xs.apply_op1(SoftmaxAnalytical { y: y.clone() })
}

/// CustomOp1 whose forward is `identity(x)` evaluated as softmax(x) via
/// the cached `y`. The key property is the `bwd`: it takes grad_res wrt
/// y and returns grad_x via the analytical softmax gradient.
///
/// The `x` fed into `apply_op1` is used as the BackpropOp anchor so the
/// autograd graph knows this op depends on x. Our bwd ignores the raw
/// `arg` and uses the cached `y` instead (more stable + cheap).
#[derive(Debug, Clone)]
struct SoftmaxAnalytical {
    y: Tensor,
}

impl CustomOp1 for SoftmaxAnalytical {
    fn name(&self) -> &'static str {
        "fused_softmax_last_dim"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        // Forward is IDENTITY of self.y — we copy the cached softmax output
        // into the output storage. Shape matches the input layout.
        let src_storage = self.y.storage_and_layout().0;
        // Extract the underlying CpuStorage; copy bytes.
        match &*src_storage {
            candle::Storage::Cpu(cpu) => Ok((cpu.clone(), layout.shape().clone())),
            _ => candle::bail!("SoftmaxAnalytical cpu_fwd called with non-cpu cached y"),
        }
    }

    fn cuda_fwd(
        &self,
        _storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let src_storage = self.y.storage_and_layout().0;
        match &*src_storage {
            candle::Storage::Cuda(cu) => {
                // Clone the CUDA storage — candle's CudaStorage implements
                // try_clone-esque via its slice.
                Ok((cu.try_clone(self.y.layout())?, layout.shape().clone()))
            }
            _ => candle::bail!("SoftmaxAnalytical cuda_fwd called with non-cuda cached y"),
        }
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // Analytical softmax Jacobian-vector product:
        //   g_x = y * (g_y - (g_y * y).sum_keepdim(-1))
        // where `y` = softmax output, `g_y` = upstream gradient.
        //
        // Use `res` (the output of this op, which IS y) over self.y —
        // they have the same values, but res is what candle's autograd
        // machinery holds.
        let y = res;
        let gy = grad_res;
        let gy_y = (gy * y)?;
        let sum = gy_y.sum_keepdim(D::Minus1)?;
        let sub = gy.broadcast_sub(&sum)?;
        let gx = (y * sub)?;
        Ok(Some(gx))
    }
}
