#include <iostream>
#include <gflags/gflags.h>
#include "onnxruntime_cxx_api.h"

DEFINE_string(weight_file, "weights/culane_18-INT32.onnx", "connect ip");
DEFINE_string(image_file, "127.0.0.1", "connect ip");

int main(int argc, char **argv)
{
  // 解析命令行参数
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::ShutDownCommandLineFlags();
  std::cout << "weight_file: " << FLAGS_weight_file << std::endl;
  std::cout << "image_file: " << FLAGS_image_file << std::endl;

  // ONNX Runtime Samples: https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx
  Ort::Env env; // Ort env
  Ort::SessionOptions session_options;

  Ort::Session session_{env, FLAGS_weight_file.c_str(), Ort::SessionOptions{nullptr}}; // CPU

  return 0;
}
