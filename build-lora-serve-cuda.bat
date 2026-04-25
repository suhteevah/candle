@echo on
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d J:\candle-src
if errorlevel 1 exit /b 1
set RUSTC_WRAPPER=
set CUDA_COMPUTE_CAP=86
set CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=link.exe
cargo build --release --example qwen-lora-serve --features cuda
exit /b %ERRORLEVEL%
