# Deployment Guide

Backend: **llama_cpp_gguf**

## Default Profile (Recommended)
- Quantization: `Q5_K_M`
- Parallel workers: `4`
- Batch config: `{'batch_size': 1, 'streaming_step_size': 3, 'max_input_length': 1024}`
- Expected metrics: `{'rougeL': 0.2982709044332347, 'latency_50_sec': 0.93560902185, 'throughput_50_samples_per_sec': 1.068822528049888, 'parallel_throughput_samples_per_sec': 7.3373026508369135}`

## Throughput Profile
- Quantization: `Q8_0`
- Parallel workers: `4`
- Batch config: `{'batch_size': 1, 'streaming_step_size': 3, 'max_input_length': 1024}`
- Expected metrics: `{'rougeL': 0.2909845795941645, 'latency_50_sec': 0.7332045656250031, 'throughput_50_samples_per_sec': 1.3638758497740304, 'parallel_throughput_samples_per_sec': 10.026194184270949}`

## Notes
- Default profile prioritizes quality for real-time production.
- Throughput profile is useful for high-volume traffic where slight quality loss is acceptable.