@echo on
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d J:\candle-src
if errorlevel 1 exit /b 1
set RUSTC_WRAPPER=
set CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=link.exe
cargo check --release --example qwen-lora-train
exit /b %ERRORLEVEL%
