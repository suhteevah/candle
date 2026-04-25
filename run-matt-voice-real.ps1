$env:CUDA_COMPUTE_CAP = '86'
Set-Location J:\candle-src

$logPath = 'J:\candle-src\logs\matt-voice-real-7b.log'
$errPath = 'J:\candle-src\logs\matt-voice-real-7b.err'

Write-Host "Launching matt-voice 7B QLoRA real fine-tune"
Write-Host "  Output dir: J:\matt-voice\adapters\matt-voice-7b-qlora-real"
Write-Host "  Log: $logPath"
Write-Host "  Err: $errPath"

$args = @(
    '--gguf', 'J:\matt-voice\models\qwen2.5-7b-q4km.gguf',
    '--tokenizer', 'C:\Users\Matt\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306\tokenizer.json',
    '--dataset', 'J:\matt-voice\training-data\matt-voice.jsonl',
    '--output-dir', 'J:\matt-voice\adapters\matt-voice-7b-qlora-real',
    '--rank', '8',
    '--alpha', '16',
    '--target-modules', 'q_proj,k_proj,v_proj,o_proj',
    '--batch-size', '1',
    '--grad-accum-steps', '4',
    '--learning-rate', '2e-4',
    '--max-steps', '1000',
    '--max-seq-len', '128',
    '--log-every', '10',
    '--save-every', '200',
    '--gradient-checkpoint'
)

$proc = Start-Process `
    -FilePath 'J:\candle-src\target\release\examples\qwen-lora-train.exe' `
    -ArgumentList $args `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError $errPath `
    -PassThru `
    -NoNewWindow

Write-Host "Started PID $($proc.Id)"
$proc.Id | Out-File -FilePath 'J:\candle-src\logs\matt-voice-real-7b.pid' -Encoding ascii
