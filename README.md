# Llama2.zig
This is a zig port of https://github.com/karpathy/llama2.c
Basic inferencing functionality supported.

## Usage
You need zig version 0.11 or newer.
```
zig build run -Doptimize=ReleaseFast -- stories15M.bin 0 128 "your prompt here"
```