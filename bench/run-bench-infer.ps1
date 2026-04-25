# run-bench-infer.ps1 — inference bench wrapper for qwen-lora-serve.
#
# Companion to run-bench.ps1 (which does training-step benches via
# qwen-lora-train --benchmark). This one does decode-throughput +
# first-token-latency benches via qwen-lora-serve --benchmark-decode N.
#
# Output JSON layout (parsed by tdpi-loop.py):
#   {
#     "commit": "<sha>", "branch": "...",
#     "preset":  "...",
#     "host":    "kokonoe", "run_at": "...",
#     "decode_tok_per_sec": 23.4,    # <-- primary metric
#     "first_token_ms":     310.5,
#     "prefill_ms":         110.2,
#     "decode_ms":          5421.0,
#     "total_ms":           5731.5,
#     "n_tokens":           128,
#     "prompt_len":         33,
#     "peak_vram_mb":       7800,
#     "label":              "..."
#   }
#
# Each preset names a flag combination on qwen-lora-serve. Adapter and
# tokenizer-config args are NOT preset-controlled — pass them via
# environment variables MODEL_GGUF, MODEL_TOKENIZER, MODEL_ADAPTER,
# MODEL_TOKENIZER_CONFIG so different model deployments can reuse
# the same preset names.

param(
    [Parameter(Mandatory)] [string] $Preset
)

$ErrorActionPreference = 'Stop'
Set-Location J:\candle-src

# --- preset definitions: extra flags after the model args ---
$presets = @{
    'infer-7b-baseline'        = @{ extra = @() }
    'infer-7b-prequant'        = @{ extra = @('--prequantize-base') }
    'infer-7b-prequant-fq'     = @{ extra = @('--prequantize-base', '--fuse-qkv') }
    'infer-7b-merge'           = @{ extra = @('--merge-adapters') }
    'infer-7b-merge-fq'        = @{ extra = @('--merge-adapters', '--fuse-qkv') }
}

if (-not $presets.ContainsKey($Preset)) {
    $known = ($presets.Keys -join ', ')
    Write-Error ('Unknown preset ' + $Preset + '. Known: ' + $known)
    exit 2
}
$presetSpec = $presets[$Preset]

# --- preflight ---
$bin = 'J:\candle-src\target\release\examples\qwen-lora-serve.exe'
if (-not (Test-Path $bin)) {
    Write-Error ('Binary missing at ' + $bin + '; build first with build-lora-serve-cuda.bat')
    exit 2
}

# Default model paths via env (override via $env:MODEL_GGUF=... before invoking).
$gguf = if ($env:MODEL_GGUF) { $env:MODEL_GGUF }
        else { 'J:\matt-voice\models\qwen2.5-7b-q4km.gguf' }
$tokenizer = if ($env:MODEL_TOKENIZER) { $env:MODEL_TOKENIZER }
        else { 'C:\Users\Matt\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306\tokenizer.json' }
$adapter = $env:MODEL_ADAPTER  # optional

$gpuLine = nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
$parts = $gpuLine -split ','
$memUsed = [int]($parts[0].Trim())
$memFree = [int]($parts[1].Trim())
$util = [int]($parts[2].Trim())
Write-Host ('[preflight] GPU: ' + $memUsed + 'MB used / ' + $memFree + 'MB free / ' + $util + 'pct util')
if ($memUsed -gt 1500) {
    Write-Error ('GPU not idle: ' + $memUsed + 'MB used. Aborting.')
    exit 3
}
# Inference needs ~5-7 GB free for 7B Q4_K_M depending on prequant flag.
if ($memFree -lt 5500) {
    Write-Error ('Only ' + $memFree + 'MB free; need 5500+ for inference bench. Aborting.')
    exit 3
}

Start-Sleep -Seconds 10

# --- run bench ---
$logDir = 'J:\candle-src\bench\results'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$commit = (git rev-parse --short HEAD).Trim()
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
$timestamp = (Get-Date -Format 'yyyyMMdd-HHmmss')
$logFile = "$logDir\$Preset-$commit-$timestamp.log"
$jsonFile = "$logDir\$Preset-$commit.json"

$env:CUDA_COMPUTE_CAP = '86'

$baseArgs = @(
    '--gguf',              $gguf,
    '--tokenizer',         $tokenizer,
    '--benchmark-decode',  '64',
    '--bench-label',       $Preset,
    '--prompt',            'Write a 60-word product blurb for a postquantum-secure SSH daemon written in Rust no_std. Be concrete and avoid filler.'
)
if ($adapter) { $baseArgs += @('--adapter', $adapter) }
if ($env:MODEL_TOKENIZER_CONFIG) {
    $baseArgs += @('--tokenizer-config', $env:MODEL_TOKENIZER_CONFIG)
}
$args = $baseArgs + $presetSpec.extra

$quoted = @($bin) + ($args | ForEach-Object {
    if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
})
$cmdLine = ($quoted -join ' ')
Write-Host ('[bench-infer] running: ' + $cmdLine)

$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
cmd /c ($cmdLine + ' > "' + $logFile + '" 2>&1')
$ErrorActionPreference = $prevEAP

# --- parse BENCH_INFER line ---
$benchLine = Get-Content $logFile | Where-Object { $_ -match '^BENCH_INFER ' } | Select-Object -Last 1
if (-not $benchLine) {
    Write-Error ('No BENCH_INFER line; log at ' + $logFile)
    exit 4
}
$benchJson = $benchLine -replace '^BENCH_INFER ', ''
$bench = $benchJson | ConvertFrom-Json

$result = [PSCustomObject]@{
    commit             = $commit
    branch             = $branch
    preset             = $Preset
    host               = $env:COMPUTERNAME
    run_at             = (Get-Date -Format 'o')
    decode_tok_per_sec = $bench.decode_tok_per_sec
    first_token_ms     = $bench.first_token_ms
    prefill_ms         = $bench.prefill_ms
    decode_ms          = $bench.decode_ms
    total_ms           = $bench.total_ms
    n_tokens           = $bench.n_tokens
    prompt_len         = $bench.prompt_len
    peak_vram_mb       = $bench.peak_vram_mb
    label              = $bench.label
    log_file           = $logFile
}
$result | ConvertTo-Json | Out-File -FilePath $jsonFile -Encoding utf8 -NoNewline
Write-Host ('[bench-infer] decode: ' + $bench.decode_tok_per_sec + ' tok/s, ttft ' + $bench.first_token_ms + ' ms, total ' + $bench.total_ms + ' ms')
Write-Host ('[bench-infer] wrote: ' + $jsonFile)
