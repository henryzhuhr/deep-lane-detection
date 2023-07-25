# 车道线检测深度学习方法
- [车道线检测深度学习方法](#车道线检测深度学习方法)
  - [任务](#任务)
  - [项目](#项目)
  - [配置环境](#配置环境)
  - [数据集准备](#数据集准备)
  - [训练](#训练)
  - [测试](#测试)
  - [部署](#部署)
    - [1. ONNX 部署](#1-onnx-部署)
    - [2. TensorRT 部署](#2-tensorrt-部署)
      - [安装 TensorRT 环境](#安装-tensorrt-环境)
      - [1. TensorRT 推理 ONNX 模型](#1-tensorrt-推理-onnx-模型)
      - [2. TensorRT 推理 Engine 模型](#2-tensorrt-推理-engine-模型)
  - [问题与解决方案](#问题与解决方案)
    - [Tensor RT Engine 转换过程中的问题](#tensor-rt-engine-转换过程中的问题)
  - [参考资料](#参考资料)



## 任务

![culane](./images/lane-define.jpg)

车道线检测 (Lane Detetction) ：需要将视频中出现的车道线检测出来。任务要求如下:
1. 查询车道线检测的相关综述，包括`传统视觉`和`深度学习`的方案
2. 通常需要检测四条车道线：左车道、自车道、右车道。
3. 需要判断双黄线、白色虚线、白色实线
4. 如果车道线终止，需要判断出结束位置
5. 车道线需要拟合成曲线，检测的车道线要求稳定，不能发生连续帧之间突变的情况

## 项目

项目是基于[Ultra Fast Lane Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)

## 配置环境

```sh
conda create -n lanedet python=3.8 -y
conda activate lanedet
```

安装[Pytorch](https://pytorch.org)依赖，需要 CUDA 11.8
```sh
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

安装其他依赖 
```shell
pip3 install -r requirements.txt
```

## 数据集准备
车道线检测常用的公开数据集:
- [CULane](https://xingangpan.github.io/projects/CULane.html)
- [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download)
- [CurveLanes](https://github.com/SoulmateB/CurveLanes):华为数据集
这里我们选择 [CULane](https://xingangpan.github.io/projects/CULane.html) 作为参考进行实验、制作自己的数据集

<!-- 作者提供了[分割工具](https://github.com/XingangPan/seg_label_generate.git)
项目实现 [PytorchAutoDrive](https://github.com/voldemortX/pytorch-auto-drive) -->

CULane 数据集处理和制作参考[CULane文档](./docs/dataset-culane.md)


无论是使用 **CULane 数据集**还是**自制数据集**，都需要设置环境变量 `$CULANEROOT`
```shell
# ~/.bashrc
export CULANEROOT=/path/to/culane
```

## 训练

首先，修改配置文件 `configs/culane.py` 中重要的参数（复制一份配置文件到 `temp/culane.py` 修改，而不是修改原始的配置文件），包括：
- `data_root` 是 CULane 数据集路径
- `log_path` 是输出的模型路径，这里默认为 `tmps`
- 训练中的超参数，比如 `epoch`,`batch_size`,`steps` 等
- `resume` 是预训练权重，如果需要加载预训练权重，可以下载官方的预训练权重: [GoogleDrive](https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view?usp=sharing)/[BaiduDrive](https://pan.baidu.com/s/19Ig0TrV8MfmFTyCvbSa4ag?pwd=w9tw)，官方的预训练权重基于 resnet18，如果希望替换成其他 backbone，需要从头训练


确认配置文件无误后启动训练脚本，需要修改脚本 `temp/culane.py` 内 
```shell
python3 train.py temp/culane.py # 单 GPU 训练
bash scripts/train-dist.sh      # 多 GPU 训练
```


## 测试
训练完成后，需要对模型进行测试，这里是对 `$CULANEROOT` 进行测试，如果只希望对单张图像进行测试，在下文提到。需要修改 `temp/culane.py` 中 `test_model`(待测试的模型权重文件) 和 `test_work_dir`(测试结果输出的目录，默认`tmps`)，然后运行
```shell
python3 test.py temp/culane.py
```
单张图像测试，需要配置的内容与上述结果相同，但是需要修改文件中 `test_img = "datasets/CULane/images/04980.jpg"`  为指定的图像
```shell
python3 infer-torch.py temp/culane.py
```

## 部署
部署分为两种方式：
1. **ONNX 部署**：将 pytorch 模型转化为 ONNX 模型，然后使用 ONNXRuntime 进行推理
2. **TensorRT 部署**：：将 pytorch 模型转化为 ONNX/Engine 模型，然后使用 TensorRT 进行推理

这里给出一种推理速度参考
|推理方式 | 平均推理时间 |
|:---:|:---:|
|Pytorch | 30.9649 ms |
|ONNXRuntime | 19.9175 ms |
|TensorRT Engine | 6.5350 ms |

TensorRT 部署在 Jetson 上需要考虑系统环境，例如 [Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) 系统中包含[JetPack 4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461):
- **OS**: Ubuntu 18.04, Linux kernel 4.9
- [**TensorRT 8.2.1**](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/quick-start-guide/index.html)
- **cuDNN 8.2.1**
- [**CUDA 10.2**](https://docs.nvidia.com/cuda/archive/10.2/cuda-toolkit-release-notes/index.html#title-new-features)



### 1. ONNX 部署

修改配置文件 `temp/culane.py` 中 `fintune` 为 pytorch 模型的权重，运行脚本后会在相同目录下生成同名的 `.onnx` 文件
```shell
python3 export.py temp/culane.py
```

得到 `.onnx` 文件后，修改推理脚本 `infer-onnx.py` 中的 onnx 模型权重 `onnx_file` 和待测试视频 `video`，然后运行如下
```shell
python3 infer-onnx.py
```

### 2. TensorRT 部署

参考 [TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html#trt_8)，部署有两种方式：
1. 使用 TensorRT 推理 ONNX 模型
2. 将 ONNX 转化为 Engine 模型并使用 TensorRT 推理



<!-- https://zhuanlan.zhihu.com/p/527238167 -->

TensorRT官方文档 ["NVIDIA Official Documentation"](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorRT Python API.](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)
- [TensorRT C++ API.](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html)
#### 安装 TensorRT 环境
必须 Ubuntu + Nvidia GPU 环境

1. 安装 CUDA、cuDNN、TensorRT，参考[官方文档](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar)，需要注册 [Nvidia 账号](https://developer.nvidia.cn/login)，并登陆

- **安装 CUDA**

  进入[CUDA 下载页面](https://developer.nvidia.com/cuda-toolkit-archive)并选择需要的版本，根据电脑的配置依次选择各项，最后选择 `deb(local)`安装方式，将出现的命令依次复制到终端中执行。

  **注意**: JetPack 4.6.1 [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)，但是 10.2 不支持 Ubuntu20，所以下载 [CUDA 11.4](https://developer.nvidia.com/cuda-11-4-0-download-archive)

- **安装 cuDNN**


  进入[cuDNN 下载页面](https://developer.nvidia.com/rdp/cudnn-archive)，根据 **CUDA 版本**和**系统架构**选择对应的版本的 **Tar**文件，下载以下内文件：
  - [`cuDNN v8.2.1, for CUDA 10.2, for Linux (x86)`](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/10.2_06072021/cudnn-10.2-linux-x64-v8.2.1.32.tgz): 安装在 Jetson Nano 上用于转换 Engine 模型
  - [`cuDNN v8.2.1, for CUDA 11.x, for Linux (x86)`](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-linux-x64-v8.2.1.32.tgz): 安装在电脑上用于转换 Engine 模型和测试，这一步是为了在电脑上编写推理代码和测试，如果推理代码已经写好，可以不用安装
  - [`cuDNN8.9.0 + CUDA 11.8 + Linux x86_64 (Tar)`](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.0/local_installers/11.8/cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz/): 最新版本

  下载后得到压缩包 `cudnn-${version}.tar.xz` ，将压缩包解压后得到同名的目录，但是8.2.1 解压后得到的是 cuda 目录，建议改名
  ```shell
  tar -xf cudnn-${version}.tar.xz
  ```

- **安装 TensorRT**

  进入[下载页面](https://developer.nvidia.com/tensorrt-getting-started)，点击 _Download Now_ 的入口，选择版本的 Tar 文件下载：
  - [`TensorRT 8.2 GA for Linux x86_64 and CUDA 11.0-5 TAR Package`](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.1/tars/tensorrt-8.2.1.8.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz)

  下载后得到压缩包 `TensorRT-${version}.tar.gz` ，将压缩包解压后得到同名的目录
  ```shell
  tar -xf TensorRT-${version}.tar.gz
  ```


在系统环境变量中添加 CUDA, cuDNN,TensorRT 相关的路径(修改为真实路径`$xxx_HOME`)。添加完成后，`source ~/.bashrc` 使环境变量生效
```shell
# ~/.bashrc
# ------ CUDA ------
CUDA_VERSION=11.8
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
# ------ cuDNN 8.2.1 ------
export CUDNN_HOME=path/to/cudnn-<version> 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_HOME/lib64 # 新版本为 lib 目录，建议自己检查一下
# ------ TensorRT 8.2.1 ------
export TENSORRT_HOME=path/to/TensorRT-<version>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_HOME/lib
export PATH=$PATH:$TENSORRT_HOME/bin
```


2. 安装 python相关环境

参考官方文档 ["Python Package Index Installation"](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip) 安装 python 的[`tensorrt`](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip)
```shell
# 根据 python 版本选择对应的 .whl 文件
python3 -m pip install --upgrade $TENSORRT_HOME/python/tensorrt-<version>.whl
python3 -m pip install --upgrade pycuda>=2020.1 # 已写入 requirements.txt
```

#### 1. TensorRT 推理 ONNX 模型

参考官方[Python API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics)


```shell
python3 -m pip install --upgrade pip
```

#### 2. TensorRT 推理 Engine 模型
参考官方文档["Exporting To ONNX From PyTorch"/"Converting ONNX To A TensorRT Engine"](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/quick-start-guide/index.html#export-from-pytorch)这种方式的部署流程是：首先将 ONNX 模型转化为 TensorRT Engine 模型，再使用 TensorRT API 推理 Engine 模型

转换过程中需要考虑到 TensorRT 支持的 ONNX 版本，具体可以参考 [TensorRT(8.2.1) 源码](https://github.com/NVIDIA/TensorRT/tree/release/8.2)中的 [`requirements.txt`](https://github.com/NVIDIA/TensorRT/blob/release/8.2/samples/python/efficientnet/requirements.txt) 文件，所以需要重新安装 ONNX 再重新转换
```shell
python3 -m pip install onnx==1.9.0 # 已写入 requirements.txt
python3 export.py temp/culane.py
```

在转换之前，确保有 `**-INT32.onnx` 模型，因为 TensorRT 支持 INT32 而不支持 INT64。如果没有，可以使用 `onnxsim` 工具进行转换
```shell
python3 -m onnxsim weights/culane_18.onnx weights/culane_18-sim.onnx
```

上述操作都可以在电脑上完成，但是如果需要在 Jetson Nano 上推理，则需要将 onnx 转换为 Engine 模型需要在 Jetson Nano 上完成，否则会出现["_不匹配设备的报错_"](#issuse-tensorrt-engine_incompatible_device)，但是后面的步骤可以在电脑上测试没有问题再在 Jetson Nano 上部署。

Jetpack 自带的库在 `/usr/src`，因此在 Jetson Nano 的系统环境变量中添加如下内容，然后`source ~/.bashrc` 使环境变量生效
```shell
# ~/.bashrc
# ------ CUDA 10.2 ------
CUDA_VERSION=10.2
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
# ------ TensorRT 8.2.1 ------
export TENSORRT_HOME=/usr/src/tensorrt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_HOME/lib
export PATH=$PATH:$TENSORRT_HOME/bin
```



使用官方提供的转换工具进行转换，如果 TensorRT 环境配置正确，将 `**-INT32.onnx` 转化为 `**-INT32.engine`。转换过程中可能出现的bug以及解决方案记录在["Tensor RT Engine 转换过程中的问题"](#TensorRT-Engine-转换过程中的问题)中，这里提供一个[输出细节参考](./docs/onnx2engine.md)
```shell
python3 export.py configs/culane.py
trtexec --verbose --fp16 \
  --onnx=weights/culane_18-INT32.onnx \
  --saveEngine=weights/culane_18-INT32.engine
```
- `--workspace=N`: Set workspace size in megabytes (default = 16)
- `--fp16`: Enable fp16 precision, in addition to fp32 (default = disabled)
- `--int8`: Enable int8 precision, in addition to fp32 (default = disabled)
- `--verbose`: Use verbose logging (default = false)
- `--exportTimes=<file>` :Write the timing results in a json file (default = disabled)
- `--exportOutput=<file>`: Write the output tensors to a json file (default = disabled)
- `--exportProfile=<file>`: Write the profile information per layer in a json file (default = disabled)

> 实际上，NV 官方提供了 [`torch2trt`](https://github.com/NVIDIA-AI-IOT/torch2trt) 转换工具可以直接完成 `PyTroch -> Engine`，但是在 Jetson Nano 上部署的时候，需要在 Jetson Nano 上完成模型转换，否则在实际使用时会出现[不匹配设备的报错](#issuse-tensorrt-engine_incompatible_device)，那么将完整的 torch 项目直接安装在 Jetson Nano 上可能会遇到很多问题，因此，这里先在电脑上完成 `PyTroch -> ONNX` 转换，然后将转换好的 `ONNX` 复制到 Jetson Nano 上使用 `trtexec` 完成 `ONNX -> Engine`  转换，可以避免在 Jetson Nano 上安装 torch 环境和项目的一些其他环境。

运行前需要安装 `pycuda`
```shell
python3 -m pip install pycuda>=2020.1
```

然后修改 `deploy/infer-trtEngine.py` 中的 `TRT_MODEL_PATH` 为 `**-INT32.engine` 的路径，运行
```shell
cd deploy
python3 infer-trtEngine.py
```

## 问题与解决方案
### Tensor RT Engine 转换过程中的问题

- 报错如下

  ```shell
  [07/13/2023-11:01:12] [E] Error[1]: [caskUtils.cpp::trtSmToCask::147] Error Code 1: Internal Error (Unsupported SM: 0x809)
  [07/13/2023-11:01:12] [E] Error[2]: [builder.cpp::buildSerializedNetwork::609] Error Code 2: Internal Error (Assertion enginePtr != nullptr failed. )
  ```
  参考 [TensorRT #2727](https://github.com/NVIDIA/TensorRT/issues/2727#issuecomment-1492809565)，`Unsupported SM` 表示该版本的 TensorRT 不支持当前的 GPU 的 SM（SM是流媒体多处理器(Streaming Multiprocessor)，RTX40系列具有与以前的GPU系列不同的SM架构），需要升级 TensorRT 版本，或者选择比如RTX3080的GPU，TensorRT 8.5.1.7以上版本支持RTX40系SM



- 报错如下<span id="issuse-tensorrt-engine_incompatible_device"></span>
  
  ```shell
  [07/14/2023-11:41:43] [TRT] [E] 6: The engine plan file is generated on an incompatible device, expecting compute 5.3 got compute 6.1, please rebuild.
  [07/14/2023-11:41:43] [TRT] [E] 4: [runtime.cpp::deserializeCudaEngine::50] Error Code 4: Internal Error (Engine deserialization failed.)
  ```
  这是由于 Engine 模型不是在 Jetson nano 上生成的，在 Jetson nano 上重新生成 Engine 模型即可

## 参考资料
- 车道线检测项目[lanedet](https://github.com/Turoad/lanedet)


- 参考 百度 Apollo 项目的[车道线检测方法](https://zhuanlan.zhihu.com/p/353339637)，使用深度学习方法进行车道线检测。
- CNN + LSTM [Robust-Lane-Detection](https://github.com/qinnzou/Robust-Lane-Detection.git)