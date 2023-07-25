import os
import shutil
import time
import datetime
from typing import Dict
import numpy as np

import torch
from torch.backends import cudnn
from model.model import parsingNet
from data.dataloader import get_train_loader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import MultiStepLR, get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger


def main():
    args, cfg = merge_config()

    device = "cpu"

    cls_num_per_lane = 18
    model = parsingNet(
        pretrained=True,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
        use_aux=cfg.use_aux,
    ).to(device)

    weights_file = cfg.finetune
    state_dict = torch.load(weights_file, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7 :]] = v
        else:
            compatible_state_dict[k] = v

    model.load_state_dict(compatible_state_dict, strict=False)
    export_onnx(model, weights_file, device=device)


def export_onnx(model: parsingNet, weights_file: str, device="cpu"):
    """
    # Export ONNX model
    see: https://pytorch.org/docs/stable/onnx.html
    """
    file_base = os.path.splitext(weights_file)[0]
    onnx_file = file_base + ".onnx"
    simplied_onnx_file = file_base + "-INT32.onnx"
    try:
        dummy_input = torch.randn(1, 3, 288, 800).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
        )

        try:
            import onnx
            onnx_model = onnx.load(onnx_file)

            # Check that the model is well formed
            onnx.checker.check_model(onnx_model)

            # Print a human readable representation of the graph
            # print(onnx.helper.printable_graph(onnx_model.graph))
            try:
                import onnxsim
                onnx_model, check = onnxsim.simplify(onnx_model)
                onnx.save(onnx_model, simplied_onnx_file)
            except Exception as e:
                print(f"Export 'ONNX'(INT32) failure: {e}")
        except Exception as e:
            print(f"Check 'ONNX':{onnx_file} failure: {e}")

        try:
            import onnxruntime as ort

            ort_session = ort.InferenceSession(onnx_file)
            print("ONNX(INT64)",onnx_file)
            print("runtime  input names:", [i.name for i in ort_session.get_inputs()],)
            print("runtime output names:", [i.name for i in ort_session.get_outputs()])
            st=time.time()
            outputs = ort_session.run(
                None,
                {"input": np.random.randn(1, 3, 288, 800).astype(np.float32)},
            )
            print("inference time:",time.time()-st)
            # print(outputs[0].shape)
        except Exception as e:
            print(f"Test 'ONNX':{onnx_file} failure: {e}")
        try:
            import onnxruntime as ort

            ort_session = ort.InferenceSession(simplied_onnx_file)
            print("ONNX(INT32)",simplied_onnx_file)
            print("runtime  input names:", [i.name for i in ort_session.get_inputs()])
            print("runtime output names:", [i.name for i in ort_session.get_outputs()])
            st=time.time()
            outputs = ort_session.run(
                None,
                {"input": np.random.randn(1, 3, 288, 800).astype(np.float32)},
            )
            print("inference time:",time.time()-st)
            # print(outputs[0].shape)
        except Exception as e:
            print(f"Test 'ONNX':{simplied_onnx_file} failure: {e}")

    except Exception as e:
            print(f"Export 'ONNX'(INT64) failure: {e}")
            
if __name__ == "__main__":
    main()