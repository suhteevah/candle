# matt-voice 7B QLoRA — overnight run on kokonoe (3070 Ti 8GB).
#
# Config: 7B Q4_K_M base + rank-8 LoRA on q/k/v/o (4 modules x 28 layers
# = 112 adapters, ~19MB trainable fp32). Seq 256 (verified to fit at
# ~7918 MB peak from earlier bench). Grad checkpointing. Save every 100
# steps for 10 intermediate checkpoints across 1000 total steps. Sends
# Telegram on completion via the same notify-telegram.sh script.
#
# Expected: ~12-15h wall-clock at ~5.7 tok/s seq=256, ~50s/step.

$env:CUDA_COMPUTE_CAP = '86'
Set-Location J:\candle-src

$logPath  = 'J:\candle-src\logs\matt-voice-overnight.log'
$errPath  = 'J:\candle-src\logs\matt-voice-overnight.err'
$pidPath  = 'J:\candle-src\logs\matt-voice-overnight.pid'
$outDir   = 'J:\matt-voice\adapters\matt-voice-7b-overnight'

Write-Host "matt-voice 7B QLoRA overnight run"
Write-Host "  log:    $logPath"
Write-Host "  err:    $errPath"
Write-Host "  output: $outDir"

$args = @(
    '--gguf',              'J:\matt-voice\models\qwen2.5-7b-q4km.gguf',
    '--tokenizer',         'C:\Users\Matt\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306\tokenizer.json',
    '--dataset',           'J:\matt-voice\training-data\matt-voice.jsonl',
    '--output-dir',        $outDir,
    '--rank',              '8',
    '--alpha',             '16',
    '--target-modules',    'q_proj,k_proj,v_proj,o_proj',
    '--batch-size',        '1',
    '--grad-accum-steps',  '4',
    '--learning-rate',     '2e-4',
    '--max-steps',         '1000',
    '--max-seq-len',       '256',
    '--log-every',         '10',
    '--save-every',        '100',
    '--gradient-checkpoint'
)

$proc = Start-Process `
    -FilePath 'J:\candle-src\target\release\examples\qwen-lora-train.exe' `
    -ArgumentList $args `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError  $errPath `
    -PassThru `
    -NoNewWindow

Write-Host "Started PID $($proc.Id) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$proc.Id | Out-File -FilePath $pidPath -Encoding ascii
