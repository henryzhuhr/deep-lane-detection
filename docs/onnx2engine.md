# ONNX 转化为 Engine 输出信息


```shell
&&&& RUNNING TensorRT.trtexec [TensorRT v8201] # trtexec --workspace=2048 --onnx=weights/culane_18-INT32.onnx --saveEngine=weights/culane_18-INT32.engine
[07/14/2023-11:52:09] [I] === Model Options ===
[07/14/2023-11:52:09] [I] Format: ONNX
[07/14/2023-11:52:09] [I] Model: weights/culane_18-INT32.onnx
[07/14/2023-11:52:09] [I] Output:
[07/14/2023-11:52:09] [I] === Build Options ===
[07/14/2023-11:52:09] [I] Max batch: explicit batch
[07/14/2023-11:52:09] [I] Workspace: 2048 MiB
[07/14/2023-11:52:09] [I] minTiming: 1
[07/14/2023-11:52:09] [I] avgTiming: 8
[07/14/2023-11:52:09] [I] Precision: FP32
[07/14/2023-11:52:09] [I] Calibration: 
[07/14/2023-11:52:09] [I] Refit: Disabled
[07/14/2023-11:52:09] [I] Sparsity: Disabled
[07/14/2023-11:52:09] [I] Safe mode: Disabled
[07/14/2023-11:52:09] [I] DirectIO mode: Disabled
[07/14/2023-11:52:09] [I] Restricted mode: Disabled
[07/14/2023-11:52:09] [I] Save engine: weights/culane_18-INT32.engine
[07/14/2023-11:52:09] [I] Load engine: 
[07/14/2023-11:52:09] [I] Profiling verbosity: 0
[07/14/2023-11:52:09] [I] Tactic sources: Using default tactic sources
[07/14/2023-11:52:09] [I] timingCacheMode: local
[07/14/2023-11:52:09] [I] timingCacheFile: 
[07/14/2023-11:52:09] [I] Input(s)s format: fp32:CHW
[07/14/2023-11:52:09] [I] Output(s)s format: fp32:CHW
[07/14/2023-11:52:09] [I] Input build shapes: model
[07/14/2023-11:52:09] [I] Input calibration shapes: model
[07/14/2023-11:52:09] [I] === System Options ===
[07/14/2023-11:52:09] [I] Device: 0
[07/14/2023-11:52:09] [I] DLACore: 
[07/14/2023-11:52:09] [I] Plugins:
[07/14/2023-11:52:09] [I] === Inference Options ===
[07/14/2023-11:52:09] [I] Batch: Explicit
[07/14/2023-11:52:09] [I] Input inference shapes: model
[07/14/2023-11:52:09] [I] Iterations: 10
[07/14/2023-11:52:09] [I] Duration: 3s (+ 200ms warm up)
[07/14/2023-11:52:09] [I] Sleep time: 0ms
[07/14/2023-11:52:09] [I] Idle time: 0ms
[07/14/2023-11:52:09] [I] Streams: 1
[07/14/2023-11:52:09] [I] ExposeDMA: Disabled
[07/14/2023-11:52:09] [I] Data transfers: Enabled
[07/14/2023-11:52:09] [I] Spin-wait: Disabled
[07/14/2023-11:52:09] [I] Multithreading: Disabled
[07/14/2023-11:52:09] [I] CUDA Graph: Disabled
[07/14/2023-11:52:09] [I] Separate profiling: Disabled
[07/14/2023-11:52:09] [I] Time Deserialize: Disabled
[07/14/2023-11:52:09] [I] Time Refit: Disabled
[07/14/2023-11:52:09] [I] Skip inference: Disabled
[07/14/2023-11:52:09] [I] Inputs:
[07/14/2023-11:52:09] [I] === Reporting Options ===
[07/14/2023-11:52:09] [I] Verbose: Disabled
[07/14/2023-11:52:09] [I] Averages: 10 inferences
[07/14/2023-11:52:09] [I] Percentile: 99
[07/14/2023-11:52:09] [I] Dump refittable layers:Disabled
[07/14/2023-11:52:09] [I] Dump output: Disabled
[07/14/2023-11:52:09] [I] Profile: Disabled
[07/14/2023-11:52:09] [I] Export timing to JSON file: 
[07/14/2023-11:52:09] [I] Export output to JSON file: 
[07/14/2023-11:52:09] [I] Export profile to JSON file: 
[07/14/2023-11:52:09] [I] 
[07/14/2023-11:52:09] [I] === Device Information ===
[07/14/2023-11:52:09] [I] Selected Device: NVIDIA Tegra X1
[07/14/2023-11:52:09] [I] Compute Capability: 5.3
[07/14/2023-11:52:09] [I] SMs: 1
[07/14/2023-11:52:09] [I] Compute Clock Rate: 0.9216 GHz
[07/14/2023-11:52:09] [I] Device Global Memory: 3955 MiB
[07/14/2023-11:52:09] [I] Shared Memory per SM: 64 KiB
[07/14/2023-11:52:09] [I] Memory Bus Width: 64 bits (ECC disabled)
[07/14/2023-11:52:09] [I] Memory Clock Rate: 0.01275 GHz
[07/14/2023-11:52:09] [I] 
[07/14/2023-11:52:09] [I] TensorRT version: 8.2.1
[07/14/2023-11:52:12] [I] [TRT] [MemUsageChange] Init CUDA: CPU +229, GPU +0, now: CPU 248, GPU 3325 (MiB)
[07/14/2023-11:52:12] [I] [TRT] [MemUsageSnapshot] Begin constructing builder kernel library: CPU 248 MiB, GPU 3354 MiB
[07/14/2023-11:52:12] [I] [TRT] [MemUsageSnapshot] End constructing builder kernel library: CPU 277 MiB, GPU 3384 MiB
[07/14/2023-11:52:12] [I] Start parsing network model
[07/14/2023-11:52:12] [I] [TRT] ----------------------------------------------------------------
[07/14/2023-11:52:12] [I] [TRT] Input filename:   weights/culane_18-INT32.onnx
[07/14/2023-11:52:12] [I] [TRT] ONNX IR version:  0.0.7
[07/14/2023-11:52:12] [I] [TRT] Opset version:    14
[07/14/2023-11:52:12] [I] [TRT] Producer name:    pytorch
[07/14/2023-11:52:12] [I] [TRT] Producer version: 2.0.1
[07/14/2023-11:52:12] [I] [TRT] Domain:           
[07/14/2023-11:52:12] [I] [TRT] Model version:    0
[07/14/2023-11:52:12] [I] [TRT] Doc string:       
[07/14/2023-11:52:12] [I] [TRT] ----------------------------------------------------------------
[07/14/2023-11:52:13] [W] [TRT] onnx2trt_utils.cpp:366: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[07/14/2023-11:52:13] [I] Finish parsing network model
[07/14/2023-11:52:13] [I] [TRT] ---------- Layers Running on DLA ----------
[07/14/2023-11:52:13] [I] [TRT] ---------- Layers Running on GPU ----------
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/conv1/Conv + /model/relu/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/maxpool/MaxPool
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer1/layer1.0/conv1/Conv + /model/layer1/layer1.0/relu/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer1/layer1.0/conv2/Conv + /model/layer1/layer1.0/Add + /model/layer1/layer1.0/relu_1/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer1/layer1.1/conv1/Conv + /model/layer1/layer1.1/relu/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer1/layer1.1/conv2/Conv + /model/layer1/layer1.1/Add + /model/layer1/layer1.1/relu_1/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer2/layer2.0/conv1/Conv + /model/layer2/layer2.0/relu/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer2/layer2.0/downsample/downsample.0/Conv
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer2/layer2.0/conv2/Conv + /model/layer2/layer2.0/Add + /model/layer2/layer2.0/relu_1/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer2/layer2.1/conv1/Conv + /model/layer2/layer2.1/relu/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer2/layer2.1/conv2/Conv + /model/layer2/layer2.1/Add + /model/layer2/layer2.1/relu_1/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer3/layer3.0/conv1/Conv + /model/layer3/layer3.0/relu/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer3/layer3.0/downsample/downsample.0/Conv
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer3/layer3.0/conv2/Conv + /model/layer3/layer3.0/Add + /model/layer3/layer3.0/relu_1/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer3/layer3.1/conv1/Conv + /model/layer3/layer3.1/relu/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer3/layer3.1/conv2/Conv + /model/layer3/layer3.1/Add + /model/layer3/layer3.1/relu_1/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer4/layer4.0/conv1/Conv + /model/layer4/layer4.0/relu/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer4/layer4.0/downsample/downsample.0/Conv
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer4/layer4.0/conv2/Conv + /model/layer4/layer4.0/Add + /model/layer4/layer4.0/relu_1/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer4/layer4.1/conv1/Conv + /model/layer4/layer4.1/relu/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /model/layer4/layer4.1/conv2/Conv + /model/layer4/layer4.1/Add + /model/layer4/layer4.1/relu_1/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /pool/Conv
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /Reshape + (Unnamed Layer* 48) [Shuffle]
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /cls/cls.0/Gemm + /cls/cls.1/Relu
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] /cls/cls.2/Gemm
[07/14/2023-11:52:13] [I] [TRT] [GpuLayer] (Unnamed Layer* 54) [Shuffle] + /Reshape_1
[07/14/2023-11:52:14] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +158, GPU +147, now: CPU 606, GPU 3701 (MiB)
[07/14/2023-11:52:16] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +241, GPU +183, now: CPU 847, GPU 3884 (MiB)
[07/14/2023-11:52:16] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
[07/14/2023-11:52:37] [I] [TRT] Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
[07/14/2023-11:55:10] [I] [TRT] Detected 1 inputs and 1 output network tensors.
[07/14/2023-11:55:10] [I] [TRT] Total Host Persistent Memory: 30784
[07/14/2023-11:55:10] [I] [TRT] Total Device Persistent Memory: 61892096
[07/14/2023-11:55:10] [I] [TRT] Total Scratch Memory: 0
[07/14/2023-11:55:10] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 113 MiB, GPU 1112 MiB
[07/14/2023-11:55:10] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 1.38893ms to assign 3 blocks to 27 nodes requiring 22118400 bytes.
[07/14/2023-11:55:10] [I] [TRT] Total Activation Memory: 22118400
[07/14/2023-11:55:10] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +0, now: CPU 1104, GPU 3281 (MiB)
[07/14/2023-11:55:10] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +0, now: CPU 1104, GPU 3281 (MiB)
[07/14/2023-11:55:10] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +256, now: CPU 0, GPU 256 (MiB)
[07/14/2023-11:55:11] [I] [TRT] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 1290, GPU 3472 (MiB)
[07/14/2023-11:55:11] [I] [TRT] Loaded engine size: 186 MiB
[07/14/2023-11:55:11] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +0, now: CPU 1290, GPU 3474 (MiB)
[07/14/2023-11:55:11] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +0, now: CPU 1290, GPU 3474 (MiB)
[07/14/2023-11:55:11] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +186, now: CPU 0, GPU 186 (MiB)
[07/14/2023-11:55:14] [I] Engine built in 184.708 sec.
[07/14/2023-11:55:14] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU -1, now: CPU 904, GPU 3324 (MiB)
[07/14/2023-11:55:14] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +0, now: CPU 904, GPU 3324 (MiB)
[07/14/2023-11:55:14] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +80, now: CPU 0, GPU 266 (MiB)
[07/14/2023-11:55:14] [I] Using random values for input input
[07/14/2023-11:55:14] [I] Created input binding for input with dimensions 1x3x288x800
[07/14/2023-11:55:14] [I] Using random values for output output
[07/14/2023-11:55:14] [I] Created output binding for output with dimensions 1x201x18x4
[07/14/2023-11:55:14] [I] Starting inference
[07/14/2023-11:55:18] [I] Warmup completed 2 queries over 200 ms
[07/14/2023-11:55:18] [I] Timing trace has 35 queries over 3.16419 s
[07/14/2023-11:55:18] [I] 
[07/14/2023-11:55:18] [I] === Trace details ===
[07/14/2023-11:55:18] [I] Trace averages of 10 runs:
[07/14/2023-11:55:18] [I] Average on 10 runs - GPU latency: 90.1627 ms - Host latency: 90.4484 ms (end to end 90.4616 ms, enqueue 7.42288 ms)
[07/14/2023-11:55:18] [I] Average on 10 runs - GPU latency: 90.1716 ms - Host latency: 90.4585 ms (end to end 90.4715 ms, enqueue 5.55878 ms)
[07/14/2023-11:55:18] [I] Average on 10 runs - GPU latency: 89.9398 ms - Host latency: 90.2251 ms (end to end 90.2382 ms, enqueue 5.77939 ms)
[07/14/2023-11:55:18] [I] 
[07/14/2023-11:55:18] [I] === Performance summary ===
[07/14/2023-11:55:18] [I] Throughput: 11.0613 qps
[07/14/2023-11:55:18] [I] Latency: min = 89.312 ms, max = 91.492 ms, mean = 90.3919 ms, median = 90.3035 ms, percentile(99%) = 91.492 ms
[07/14/2023-11:55:18] [I] End-to-End Host Latency: min = 89.3254 ms, max = 91.5049 ms, mean = 90.4048 ms, median = 90.3167 ms, percentile(99%) = 91.5049 ms
[07/14/2023-11:55:18] [I] Enqueue Time: min = 1.54553 ms, max = 9.62085 ms, mean = 6.19246 ms, median = 6.2948 ms, percentile(99%) = 9.62085 ms
[07/14/2023-11:55:18] [I] H2D Latency: min = 0.274475 ms, max = 0.288208 ms, mean = 0.277587 ms, median = 0.276855 ms, percentile(99%) = 0.288208 ms
[07/14/2023-11:55:18] [I] GPU Compute Time: min = 89.0273 ms, max = 91.2045 ms, mean = 90.1061 ms, median = 90.0183 ms, percentile(99%) = 91.2045 ms
[07/14/2023-11:55:18] [I] D2H Latency: min = 0.00683594 ms, max = 0.00881958 ms, mean = 0.008238 ms, median = 0.00830078 ms, percentile(99%) = 0.00881958 ms
[07/14/2023-11:55:18] [I] Total Host Walltime: 3.16419 s
[07/14/2023-11:55:18] [I] Total GPU Compute Time: 3.15371 s
[07/14/2023-11:55:18] [I] Explanations of the performance metrics are printed in the verbose logs.
[07/14/2023-11:55:18] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8201] # trtexec --workspace=2048 --onnx=weights/culane_18-INT32.onnx --saveEngine=weights/culane_18-INT32.engine
```