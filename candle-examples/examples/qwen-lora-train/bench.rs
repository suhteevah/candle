//! Microbenchmark harness for qwen-lora-train.
//!
//! # Why this exists
//!
//! The previous optimization pass burned hours on "is this faster?" guesses
//! without a controlled way to answer. This module establishes a single
//! reproducible measurement protocol so every future optimization can be
//! accepted or rejected based on numbers.
//!
//! # What it measures
//!
//! - **tokens_per_sec**: throughput = (measured_steps * micro_batches_per_step
//!   * micro_batch_tokens) / measured_wall_clock_seconds. Uses the full
//!   `[B * L]` token count per micro-batch, not just loss-mask positions —
//!   we want to measure compute, not data efficiency.
//! - **step_time_ms_median / p95**: per-optimizer-step latency.
//!   Median is the stable central tendency; p95 catches backward spikes.
//! - **peak_vram_mb**: maximum `memory.used` sampled via nvidia-smi during
//!   the measured run (plus start/end snapshots). Not perfectly peaked
//!   since we sample at step boundaries, but tight enough to detect
//!   1-2GB regressions.
//! - **mean_gpu_util**: mean of nvidia-smi `utilization.gpu` across samples.
//!   Low values (<60%) usually mean CPU bottleneck OR allocator thrash.
//!
//! # Protocol (enforced by default config below)
//!
//! 1. Pre-tokenize a fixed pool of examples upfront. Removes dataloader
//!    variance from the measurement.
//! 2. Warmup: 3 optimizer steps with NO measurement (stabilizes caches,
//!    warms up kernel autotuning).
//! 3. Measured: 20 optimizer steps with per-step timing.
//! 4. VRAM sampled at start, after warmup, after every 2 measured steps,
//!    and at end. Max of these.
//! 5. Results printed on a SINGLE LINE in stable JSON-compatible form for
//!    easy `tail | awk`-ing across runs.
//!
//! # How to read the output
//!
//! A line like:
//!
//! ```text
//! BENCH {"tokens_per_sec":1234.5,"step_ms_median":85.3,"step_ms_p95":91.2,"peak_vram_mb":7840,"mean_gpu_util":82.1,"cfg":"rank=8,targets=q_proj,v_proj,seq=128,ga=4,gc=true"}
//! ```
//!
//! Compare tokens_per_sec and peak_vram_mb across runs. A legitimate
//! optimization moves one up or one down without regressing the other.

use anyhow::Result;
use std::process::Command;
use std::time::Instant;

/// Benchmark configuration. Defaults match the numbers baked into the
/// protocol — override only with specific reason.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub warmup_steps: usize,
    pub measure_steps: usize,
    /// Whether to call nvidia-smi for VRAM/util stats. Off on systems
    /// without CUDA to avoid spurious warnings in output.
    pub sample_nvidia_smi: bool,
    /// Config label that gets embedded in the output line. Helps when
    /// grepping across many runs.
    pub cfg_label: String,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            warmup_steps: 3,
            measure_steps: 20,
            sample_nvidia_smi: true,
            cfg_label: "".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub tokens_per_sec: f64,
    pub step_time_ms_median: f64,
    pub step_time_ms_p95: f64,
    pub peak_vram_mb: Option<u64>,
    pub mean_gpu_util: Option<f64>,
    pub cfg_label: String,
}

impl BenchResult {
    /// Render the single-line BENCH output format.
    pub fn to_single_line(&self) -> String {
        let peak = self
            .peak_vram_mb
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".into());
        let util = self
            .mean_gpu_util
            .map(|v| format!("{:.1}", v))
            .unwrap_or_else(|| "null".into());
        format!(
            r#"BENCH {{"tokens_per_sec":{:.1},"step_ms_median":{:.2},"step_ms_p95":{:.2},"peak_vram_mb":{},"mean_gpu_util":{},"cfg":"{}"}}"#,
            self.tokens_per_sec,
            self.step_time_ms_median,
            self.step_time_ms_p95,
            peak,
            util,
            self.cfg_label.replace('"', "'")
        )
    }
}

/// Run a benchmark using a caller-supplied optimizer-step closure.
///
/// `step_fn` does one complete optimizer step: forward + loss + backward
/// + optim.step + any per-step bookkeeping. It returns the number of
/// tokens processed by that step (typically `batch * grad_accum * seq_len`).
///
/// The harness calls `step_fn` for `warmup_steps + measure_steps` total
/// invocations, timing only the measured subset.
pub fn run<F>(cfg: &BenchConfig, mut step_fn: F) -> Result<BenchResult>
where
    F: FnMut(usize) -> Result<usize>,
{
    // Initial VRAM snapshot (just before warmup).
    let mut vram_samples: Vec<u64> = Vec::new();
    let mut util_samples: Vec<u64> = Vec::new();
    if cfg.sample_nvidia_smi {
        if let Some((v, u)) = sample_nvidia_smi() {
            vram_samples.push(v);
            util_samples.push(u);
        }
    }

    // Warmup — no timing.
    for i in 0..cfg.warmup_steps {
        let _tokens = step_fn(i)?;
    }

    // Post-warmup sample.
    if cfg.sample_nvidia_smi {
        if let Some((v, u)) = sample_nvidia_smi() {
            vram_samples.push(v);
            util_samples.push(u);
        }
    }

    // Measured loop. Collect per-step wall times; total tokens.
    let mut step_times_ms: Vec<f64> = Vec::with_capacity(cfg.measure_steps);
    let mut total_tokens: u64 = 0;
    let measure_start = Instant::now();
    for i in 0..cfg.measure_steps {
        let step_start = Instant::now();
        let tokens = step_fn(cfg.warmup_steps + i)?;
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        step_times_ms.push(step_ms);
        total_tokens += tokens as u64;

        // Sample VRAM every 2 steps to limit overhead. nvidia-smi spawn is
        // ~20ms on Windows, so sampling every step would pollute timing.
        if cfg.sample_nvidia_smi && i % 2 == 0 {
            if let Some((v, u)) = sample_nvidia_smi() {
                vram_samples.push(v);
                util_samples.push(u);
            }
        }
    }
    let measure_elapsed_s = measure_start.elapsed().as_secs_f64();

    // Final sample.
    if cfg.sample_nvidia_smi {
        if let Some((v, u)) = sample_nvidia_smi() {
            vram_samples.push(v);
            util_samples.push(u);
        }
    }

    // Reduce.
    step_times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = step_times_ms[step_times_ms.len() / 2];
    let p95_idx = ((step_times_ms.len() as f64) * 0.95) as usize;
    let p95 = step_times_ms[p95_idx.min(step_times_ms.len() - 1)];

    let tokens_per_sec = if measure_elapsed_s > 0.0 {
        total_tokens as f64 / measure_elapsed_s
    } else {
        0.0
    };

    let peak_vram_mb = vram_samples.iter().copied().max();
    let mean_gpu_util = if !util_samples.is_empty() {
        Some(util_samples.iter().sum::<u64>() as f64 / util_samples.len() as f64)
    } else {
        None
    };

    Ok(BenchResult {
        tokens_per_sec,
        step_time_ms_median: median,
        step_time_ms_p95: p95,
        peak_vram_mb,
        mean_gpu_util,
        cfg_label: cfg.cfg_label.clone(),
    })
}

/// One nvidia-smi probe. Returns `(memory_used_mb, utilization_percent)`
/// or None if the shell-out fails (e.g. no CUDA).
fn sample_nvidia_smi() -> Option<(u64, u64)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    // Line like: "1023, 87"
    let line = stdout.lines().next()?;
    let mut parts = line.split(',').map(|s| s.trim());
    let mem: u64 = parts.next()?.parse().ok()?;
    let util: u64 = parts.next()?.parse().ok()?;
    Some((mem, util))
}
