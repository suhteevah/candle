$env:CUDA_COMPUTE_CAP = '86'
Set-Location J:\candle-src
& .\target\release\examples\qwen-lora-train.exe `
    --gguf 'J:\matt-voice\models\qwen2.5-7b-q4km.gguf' `
    --tokenizer 'C:\Users\Matt\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306\tokenizer.json' `
    --dataset 'J:\matt-voice\training-data\matt-voice.jsonl' `
    --output-dir 'J:\tmp\bench-noop' `
    --rank 8 --alpha 16 --target-modules q_proj,v_proj `
    --batch-size 1 --grad-accum-steps 4 `
    --max-seq-len 128 --gradient-checkpoint `
    --benchmark --bench-label '7b-q4km-frope-s128' 2>&1 |
    Tee-Object -FilePath J:\candle-src\logs\7b-frope-s128.log
