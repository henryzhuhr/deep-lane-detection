import os
import time
import cv2
import numpy as np

import torch
from torchvision import transforms
import tqdm
from model.model import parsingNet

from utils.common import merge_config


test_img = "datasets/CULane/images/04980.jpg"


class NormalizeValue:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


COLOR_LIST = [
    (b, g, r) for (color_name, (r, g, b)) in {
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
    }.items()
]
cls_num_per_lane = 18
row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

img_transforms = transforms.Compose(
    [
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


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

    weights_file = cfg.test_model

    state_dict = torch.load(weights_file, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7 :]] = v
        else:
            compatible_state_dict[k] = v

    model.load_state_dict(compatible_state_dict, strict=False)

    

    frame = cv2.imread(test_img)

    time_list = []
    pbar= tqdm.tqdm(range(100))
    for i in pbar:
        st = time.time()
        img_h, img_w, _ = frame.shape

        img: cv2.Mat = cv2.resize(frame, (800, 288))

        img = img / 255.0
        img = (img - NormalizeValue.mean) / NormalizeValue.std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        images = torch.from_numpy(img).float()

        


        with torch.no_grad():
            out = model.forward(images.to(device))
            out = out[0].data.cpu().numpy()
            
            

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        # flip_updown
        out = out[:, ::-1, :]   

        # relative localization
        import scipy.special
        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out = np.argmax(out, axis=0)
        loc[out == cfg.griding_num] = 0
        out = loc      # shape [18,4]

        four_lanes = []
        for i in range(out.shape[1]):
            color = COLOR_LIST[i]
            lane_i = []
            if np.sum(out[:, i] != 0) > 2:
                for k in range(out.shape[0]):
                    if out[k, i] > 0:
                        ppp = (
                            int(out[k, i] * col_sample_w * img_w / 800) - 1,
                            int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                        )
                        cv2.circle(frame, ppp, 5, color, -1)
                        lane_i.append(ppp)
            four_lanes.append(lane_i)
        infer_time = (time.time() - st) * 1000
        pbar.set_description("time: %.4f ms" % infer_time)
        time_list.append(infer_time)
        cv2.imwrite(os.path.join(cfg.test_work_dir,"test.jpg"), frame)
    print("avg time: %.4f ms" % np.mean(time_list))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == "__main__":
    main()