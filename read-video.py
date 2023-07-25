import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor

torch.backends.cudnn.benchmark = True


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


color_map = {
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
}

color_list = [(b, g, r) for (color_name, (r, g, b)) in color_map.items()]


def main():
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(
        pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4), use_aux=False
    ).cuda()
    
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7 :]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose(
        [
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # 加载视频
    capture = cv2.VideoCapture("example.mp4")
    # capture = cv2.VideoCapture(0)

    out_video = cv2.VideoWriter(
        "example-out.mp4",
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=capture.get(cv2.CAP_PROP_FPS),
        frameSize=(
            frame_width := int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            frame_height := int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    os.makedirs(outdir := "outs", exist_ok=True)
    i = 0
    row_anchor = culane_row_anchor if cfg.dataset == 'CULane' else tusimple_row_anchor
    while True:
        ret, frame = capture.read() # frame: (320, 800, 3)  [ H, W, C ]

        if not ret:
            break
        img_h, img_w, _ = frame.shape
        # colour_filtered = filter_colors(frame)
        img = cv2.resize(frame, (800, 288))
        images = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        images = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(images) / 255.

        with torch.no_grad():
            out = net.forward(images)

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = softmax(out_j[:-1, :, :])
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc # shape [18,4]
                    # 拟合车道线，`np.polyfit` https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/issues/57

        four_lanes = []
        for i in range(out_j.shape[1]):
            color = color_list[i]
            lane_i = []
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (
                            int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                            int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                        )
                        cv2.circle(frame, ppp, 5, color, -1)
                        lane_i.append(ppp)
            four_lanes.append(lane_i)

        for i, lane_i in enumerate(four_lanes):
            color = color_list[i]
            lane_x, lane_y = [], []
            min_y = img_h

            points = np.array([lane_i], dtype=np.int32) # (1, 10, 2)
            if points.shape[1] > 2:
                cv2.fillPoly(frame, points, color)
            continue

            for (x, y) in lane_i:
                min_y = y if y < min_y else min_y
                lane_x.append(x)
                lane_y.append(y)
            if len(lane_x) > 2:
                cv2.fillPoly(frame, np.int_([x, y]), color)
                # z1 = np.polyfit(lane_y, lane_x, 2)
                # p1 = np.poly1d(z1)
                # for y in range(min_y, img_h, 2):
                #     x = int(p1(y))
                #     cv2.circle(frame, (x, y), 1, color, -1)

        # img_h, img_w, _ = frame.shape
        for i in range(out_j.shape[1]):
            cv2.circle(frame, (int(img_w / 2) + 50 * (i * 2 - 3), int(img_h * 0.01) + 10), 10, color_list[i], -1)
        cv2.imwrite(os.path.join(outdir, f"frame.jpg"), frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

        out_video.write(frame)
    capture.release()
    out_video.release() #资源释放
    cv2.destroyAllWindows()


def filter_colors(image):
    # Filter white pixels
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([0, 80, 80])
    upper_yellow = np.array([110, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    return image2


if __name__ == "__main__":
    main()