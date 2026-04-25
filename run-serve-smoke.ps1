Set-Location J:\candle-src
& .\target\release\examples\qwen-lora-serve.exe `
    --gguf 'J:\matt-voice\models\matt-voice-1.5b-q4_0.gguf' `
    --tokenizer 'C:\Users\Matt\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306\tokenizer.json' `
    --prompt 'Hello, how are you' `
    --max-tokens 4 `
    --temperature 0 `
    --cpu 2>&1 | Tee-Object -FilePath J:\candle-src\logs\serve-smoke.log
