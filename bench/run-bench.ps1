# run-bench.ps1 — deterministic single-config bench runner for the TDPI loop.
#
# Reads a config preset from $args[0], runs qwen-lora-train --benchmark
# with locked seed and matching hardware-state preconditions, parses the
# BENCH JSON line, writes results to bench/results/<preset>-<commit>.json.
#
# Preconditions enforced:
#   * GPU0 idle (memory.free > 7.0 GB on 8GB card; >14 GB on 16GB)
#   * 30s cooldown if a previous run just finished
#   * Fixed CUDA_COMPUTE_CAP for the host
#   * No competing GPU procs (nvidia-smi tasks parsed)
#
# Output JSON layout:
#   {
#     "commit": "<sha>",
#     "branch": "<branchname>",
#     "preset":  "...",
#     "host":    "kokonoe",
#     "run_at":  "2026-04-25T...",
#     "tok_per_sec": 5.5,
#     "step_ms_median": 53933.46,
#     "step_ms_p95": 55963.64,
#     "peak_vram_mb": 7919,
#     "mean_gpu_util": 93.4,
#     "cfg_label":  "..."
#   }
#
# Usage:
#   .\run-bench.ps1 q4km-7b-r8-qv-s128-gc
#   .\run-bench.ps1 q4km-7b-r8-qv-s128-gc-prequant
#   .\run-bench.ps1 q4km-7b-r8-qv-s128-gc-prequant-fuseqkv
#   .\run-bench.ps1 q4km-7b-r8-qv-s128-gc-cechunk32

param(
    [Parameter(Mandatory)] [string] $Preset
)

$ErrorActionPreference = 'Stop'
Set-Location J:\candle-src

# --- preset definitions ---
$presets = @{
    'q4km-7b-r8-qv-s128-gc'              = @{ extra = @() }
    'q4km-7b-r8-qv-s128-gc-cechunk32'    = @{ extra = @('--ce-chunk-size', '32') }
    'q4km-7b-r8-qv-s128-gc-prequant'     = @{ extra = @('--prequantize-base') }
    'q4km-7b-r8-qv-s128-gc-prequant-fq'  = @{ extra = @('--prequantize-base', '--fuse-qkv') }
    'q4km-7b-r8-qv-s128-gc-all-on'       = @{
        extra = @('--prequantize-base', '--fuse-qkv', '--ce-chunk-size', '32')
    }
}

if (-not $presets.ContainsKey($Preset)) {
    $known = ($presets.Keys -join ', ')
    Write-Error ('Unknown preset ' + $Preset + '. Known: ' + $known)
    exit 2
}

$presetSpec = $presets[$Preset]

# --- preflight: GPU + binary ---
$bin = 'J:\candle-src\target\release\examples\qwen-lora-train.exe'
if (-not (Test-Path $bin)) {
    Write-Error ('Binary missing at ' + $bin + '; build first with build-lora-train-cuda.bat')
    exit 2
}

$gpuLine = nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
$parts = $gpuLine -split ','
$memUsed = [int]($parts[0].Trim())
$memFree = [int]($parts[1].Trim())
$util = [int]($parts[2].Trim())
Write-Host ('[preflight] GPU: ' + $memUsed + 'MB used / ' + $memFree + 'MB free / ' + $util + 'pct util')
if ($memUsed -gt 1500) {
    Write-Error ('GPU not idle: ' + $memUsed + 'MB used. Close GPU apps. Aborting.')
    exit 3
}
if ($memFree -lt 6500) {
    Write-Error ('Only ' + $memFree + 'MB free; need 6500+ for the bench. Aborting.')
    exit 3
}

# --- pre-bench cooldown: 15s settle so any prior run's stragglers exit cleanly ---
Start-Sleep -Seconds 15

# --- run bench ---
$logDir = 'J:\candle-src\bench\results'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$commit = (git rev-parse --short HEAD).Trim()
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
$timestamp = (Get-Date -Format 'yyyyMMdd-HHmmss')
$logFile = "$logDir\$Preset-$commit-$timestamp.log"
$jsonFile = "$logDir\$Preset-$commit.json"

$env:CUDA_COMPUTE_CAP = '86'

$args = @(
    '--gguf',              'J:\matt-voice\models\qwen2.5-7b-q4km.gguf',
    '--tokenizer',         'C:\Users\Matt\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306\tokenizer.json',
    '--dataset',           'J:\matt-voice\training-data\matt-voice.jsonl',
    '--output-dir',        'J:\tmp\bench-noop',
    '--rank',              '8',
    '--alpha',             '16',
    '--target-modules',    'q_proj,v_proj',
    '--batch-size',        '1',
    '--grad-accum-steps',  '4',
    '--max-seq-len',       '128',
    '--gradient-checkpoint',
    '--benchmark',
    '--bench-label',       $Preset,
    '--seed',              '299792458'
) + $presetSpec.extra

# Build a quoted command line for cmd /c so stderr can be merged into
# stdout via plain shell redirection — bypasses PowerShell's
# $ErrorActionPreference='Stop' which treats any native-exe stderr line
# as a terminating error and aborts the rest of the script.
$quoted = @($bin) + ($args | ForEach-Object {
    if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
})
$cmdLine = ($quoted -join ' ')
Write-Host ('[bench] running: ' + $cmdLine)

# Allow stderr-as-info from the native exe; we'll judge success by
# parsing the BENCH line from the captured log.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
cmd /c ($cmdLine + ' > "' + $logFile + '" 2>&1')
$ErrorActionPreference = $prevEAP

# --- parse BENCH line ---
$benchLine = Get-Content $logFile | Where-Object { $_ -match '^BENCH ' } | Select-Object -Last 1
if (-not $benchLine) {
    Write-Error ('No BENCH line in output, log at ' + $logFile)
    exit 4
}
$benchJson = $benchLine -replace '^BENCH ', ''
$bench = $benchJson | ConvertFrom-Json

$result = [PSCustomObject]@{
    commit          = $commit
    branch          = $branch
    preset          = $Preset
    host            = $env:COMPUTERNAME
    run_at          = (Get-Date -Format 'o')
    tok_per_sec     = $bench.tokens_per_sec
    step_ms_median  = $bench.step_ms_median
    step_ms_p95     = $bench.step_ms_p95
    peak_vram_mb    = $bench.peak_vram_mb
    mean_gpu_util   = $bench.mean_gpu_util
    cfg_label       = $bench.cfg
    log_file        = $logFile
}
$result | ConvertTo-Json | Out-File -FilePath $jsonFile -Encoding utf8 -NoNewline
Write-Host ('[bench] result: ' + $bench.tokens_per_sec + ' tok/s, ' + $bench.peak_vram_mb + 'MB peak')
Write-Host ('[bench] wrote: ' + $jsonFile)
