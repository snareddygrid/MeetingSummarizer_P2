# Quantization Benchmark Report

Recommended quantization: **Q8_0**
Recommended parallel setup: **4 processes** on Q8_0 (throughput=10.0262 samples/sec)
Fastest overall parallel setup: **4 processes** on Q8_0 (throughput=10.0262 samples/sec)
Quantization backend: **llama_cpp_gguf**

## Model Comparison

| Model | ROUGE-L | Latency@50 (s) | Throughput@50 (samples/s) | Streaming ΔROUGE-L | Score |
|---|---:|---:|---:|---:|---:|
| Q8_0 | 0.2910 | 0.7332 | 1.3639 | 0.0000 | 23.1303 |
| Q5_K_M | 0.2983 | 0.9356 | 1.0688 | 0.0000 | 21.5398 |
| Q4_K_M | 0.2981 | 1.0236 | 0.9770 | 0.0000 | 20.5477 |