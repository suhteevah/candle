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
use candle::{CpuStorage, CustomOp1, DType, Layout, Module, Result, Shape, Tensor, D};

/// Softmax over the last dimension with a compact analytical backward.
///
/// Drop-in replacement for `candle_nn::ops::softmax(xs, D::Minus1)?` on
/// the training hot path. At inference time it's roughly equivalent.
pub fn fused_softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    let y = candle_nn::ops::softmax(xs, D::Minus1)?;
    // Detach y before caching so the composed forward's 5 intermediates
    // can drop immediately (they're only reachable via y's BackpropOp
    // chain; detach severs that, Rust RAII does the rest).
    let y_cached = y.detach();
    drop(y); // explicit: y's BackpropOp → composed intermediates die now
    xs.apply_op1(SoftmaxAnalytical { y: y_cached })
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

// ============================================================================
// Stage 2: Fused RMSNorm
// ============================================================================
//
// Composed reference: `candle_nn::ops::rms_norm_slow` does sqr + sum_keepdim
// + div + sqrt + broadcast_div + broadcast_mul — 5-6 intermediate tensors.
//
// Our fused op computes the same forward value (via the composed path, once)
// but registers a CustomOp1 that caches only `x` (the input) + α (the
// frozen weight) + ε for the analytical backward. No intermediates retained.
//
// Analytical backward (frozen α, so no grad wrt α):
//   Let r_b = 1 / sqrt( mean_j(x_j²) + ε )   (one scalar per batch position)
//   y_i = r_b * α_i * x_i
//   ∂L/∂x_i = r_b * α_i * ∂L/∂y_i
//             − r_b³ * x_i * mean_j( α_j * x_j * ∂L/∂y_j )
//
// Cache during training: only `x` and `α` (and `ε` is just f32).

/// Fused RMSNorm over the last dim, with α (weight) assumed frozen.
///
/// `alpha` should be shape [hidden]; broadcasts over all leading dims of x.
pub fn fused_rms_norm(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let y = candle_nn::ops::rms_norm_slow(x, alpha, eps)?;
    // Detach the cached forward output so composed intermediates drop.
    let y_cached = y.detach();
    drop(y);
    // x we keep live (it's the input; its id is how the outer walker
    // accumulates our returned grad_x, so we can't detach this one).
    // alpha is frozen (not a Var), cheap clone.
    let op = RmsNormAnalytical {
        y: y_cached,
        x: x.clone(),
        alpha: alpha.clone(),
        eps,
    };
    x.apply_op1(op)
}

#[derive(Debug, Clone)]
struct RmsNormAnalytical {
    y: Tensor,
    x: Tensor,
    alpha: Tensor,
    eps: f32,
}

impl CustomOp1 for RmsNormAnalytical {
    fn name(&self) -> &'static str {
        "fused_rms_norm"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let src_storage = self.y.storage_and_layout().0;
        match &*src_storage {
            candle::Storage::Cpu(cpu) => Ok((cpu.clone(), layout.shape().clone())),
            _ => candle::bail!("RmsNormAnalytical cpu_fwd called with non-cpu cached y"),
        }
    }

    fn cuda_fwd(
        &self,
        _storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let src_storage = self.y.storage_and_layout().0;
        match &*src_storage {
            candle::Storage::Cuda(cu) => Ok((cu.try_clone(self.y.layout())?, layout.shape().clone())),
            _ => candle::bail!("RmsNormAnalytical cuda_fwd called with non-cuda cached y"),
        }
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // Work in fp32 internally for numerical stability; cast back at the
        // end to the input x's dtype.
        let orig_dtype = self.x.dtype();
        let x = if orig_dtype == DType::F32 {
            self.x.clone()
        } else {
            self.x.to_dtype(DType::F32)?
        };
        let alpha = if self.alpha.dtype() == DType::F32 {
            self.alpha.clone()
        } else {
            self.alpha.to_dtype(DType::F32)?
        };
        let gy = if grad_res.dtype() == DType::F32 {
            grad_res.clone()
        } else {
            grad_res.to_dtype(DType::F32)?
        };

        let hidden = x.dim(D::Minus1)? as f64;
        // r_b keepdim [..., 1]
        let mean_sq = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden)?;
        let r = (mean_sq + self.eps as f64)?.sqrt()?.recip()?;

        // term1: r_b * α_i * gy_i
        let alpha_gy = gy.broadcast_mul(&alpha)?;
        let term1 = alpha_gy.broadcast_mul(&r)?;

        // term2: r_b³ * x_i * mean_j(α_j * x_j * gy_j)
        let alpha_x_gy = alpha_gy.mul(&x)?;
        let mean_agx = (alpha_x_gy.sum_keepdim(D::Minus1)? / hidden)?;
        let r3 = (&r * &r)?.mul(&r)?;
        let scale = r3.broadcast_mul(&mean_agx)?;
        let term2 = x.broadcast_mul(&scale)?;

        let gx = (term1 - term2)?;
        let gx = if gx.dtype() != orig_dtype {
            gx.to_dtype(orig_dtype)?
        } else {
            gx
        };
        Ok(Some(gx))
    }
}

// Silence unused-import warning from Module import used for signature
#[allow(dead_code)]
fn _module_fence<M: Module>(_: M) {}
