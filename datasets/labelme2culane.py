import os
import json
import shutil
import sys
import cv2
import numpy as np

import tqdm

from common import CULaneDIR
from common import SUPPORTED_IMAGE_FORMATS, COLOR_MAP
from common import get_culane_root

IS_RM_LEAGACY = False # 是否删除旧的文件夹，包括 labels_dir, list_dir, visual_dir
LANE_DEFINE = [1, 2, 3, 4]


class LANE_LABEL_COLOR:
    y_seq = (70, 120, 60)
    black = (0, 0, 0)
    dark = (139, 139, 0)


COLOR_LIST = list(COLOR_MAP.values())


def main():
    culane_root = get_culane_root()
    culane_root = "/Users/henryzhu/datasets/CULane-custom"
    raw_images_dir = os.path.join(culane_root, CULaneDIR.raw_images_dir)
    images_dir, labels_dir, list_dir, visual_dir = create_dir(culane_root)

    train_gt_file = open(os.path.join(list_dir, "train_gt.txt"), "w")

    for video_name in os.listdir(raw_images_dir):
        print(raw_images_dir, video_name)
        # 如果不是目录，跳过
        if not os.path.isdir(raw_images_dir) or video_name == '.DS_Store':
            continue

        video_image_dir = os.path.join(raw_images_dir, video_name)
        print("processing video: ", video_image_dir)

        # video_visual_dir = os.path.join(visual_dir, video_name)
        # os.makedirs(video_visual_dir, exist_ok=True)

        train_list = []
        pbar = tqdm.tqdm(os.listdir(video_image_dir))
        for file in pbar:
            file_name, file_ext = os.path.splitext(file)
            if file_ext in SUPPORTED_IMAGE_FORMATS:
                json_file = os.path.join(video_image_dir, file_name + ".json")
                if os.path.exists(json_file):
                    pbar.set_description(f"Processing {file}")

                    raw_dict = convert_label(json_file)
                    raw_lanes_dict = {
                        label:
                            sorted(
                                lane,
                                key=lambda x: x[1],         # sort by y of (x,y)
                                reverse=True,               # True for line points from top to bottom
                            )
                        for label, lane in raw_dict.items()
                    }

                    img = cv2.imread(os.path.join(video_image_dir, file))
                    img_h, img_w, _ = img.shape
                    crop_img_h = int(img_w * 540 / 800) # 800x288 比例裁切图像下半部分
                    # img = img[img_h - crop_img_h : img_h, :]# TODO 是否裁切下半部分
                    raw_img = img.copy()

                    lanes_dict = {}
                    for label, lane in raw_lanes_dict.items():
                        new_lane = []
                        for (x, y) in lane:
                            # TODO 是否裁切下半部分
                            # if y > img_h - crop_img_h:
                            #     new_lane.append((x, y - (img_h - crop_img_h))) # 裁切后高度也要对应减去
                            new_lane.append((x, y)) # TODO 是否裁切下半部分

                        lanes_dict[label] = new_lane

                    has_lane = {str(id): 0 for id in LANE_DEFINE}

                    with open(os.path.join(images_dir, f"{file_name}.lines.txt"), "w") as f:
                        for i, (label, lane) in enumerate(lanes_dict.items()):
                            color = COLOR_LIST[i % len(COLOR_LIST)]
                            color = (color[2], color[1], color[0])
                            if len(lane) > 0:
                                has_lane[label] = 1
                                for (x, y) in lane:
                                    cv2.circle(img, (int(x), int(y)), 5, color, -1)
                                    f.write(f"{x} {y} ")
                                f.write("\n")
                    cv2.imwrite(os.path.join(visual_dir, file), img)
                    cv2.imwrite(os.path.join(images_dir, file), raw_img)

                    # 创建标签
                    label_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    for i, (label, lane) in enumerate(lanes_dict.items()):
                        if len(lane) > 0:
                            for j in range(len(lane) - 1):
                                cv2.line(
                                    label_img,
                                    (int(lane[j][0]), int(lane[j][1])),
                                    (int(lane[j + 1][0]), int(lane[j + 1][1])),
                                    LANE_DEFINE[i],
                                    5,
                                )
                    cv2.imwrite(os.path.join(labels_dir, f"{file_name}.png"), label_img)

                    label_item = [
                        f"{CULaneDIR.images_dir}/{file_name}.jpg", # based on culane_root
                        f"{CULaneDIR.labels_dir}/{file_name}.png", # no os.path.join for win may generate '\'
                        "%d %d %d %d" % (
                            has_lane["1"],
                            has_lane["2"],
                            has_lane["3"],
                            has_lane["4"],
                        )
                    ]
                    train_gt_file.write(" ".join(label_item) + "\n")
                else:
                    print("\033[1;33m Not found %s\033[0m" % json_file)
    train_gt_file.close()


def create_dir(culane_root: str):
    assert os.path.exists(culane_root), f"CULANEROOT not found in :{culane_root}"
    raw_images_path = os.path.join(culane_root, CULaneDIR.raw_images_dir)
    assert os.path.exists(raw_images_path), f"raw images not found in :{raw_images_path}"

    # 保存图片的目录
    images_dir = os.path.join(culane_root, CULaneDIR.images_dir)
    if os.path.exists(images_dir) and IS_RM_LEAGACY:
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)

    # 保存标签的目录
    labels_dir = os.path.join(culane_root, CULaneDIR.labels_dir)
    if os.path.exists(labels_dir) and IS_RM_LEAGACY:
        shutil.rmtree(labels_dir)
    os.makedirs(labels_dir, exist_ok=True)

    # 保存训练列表的目录
    list_dir = os.path.join(culane_root, CULaneDIR.list_dir)
    if os.path.exists(list_dir) and IS_RM_LEAGACY:
        shutil.rmtree(list_dir)
    os.makedirs(list_dir, exist_ok=True)

    # 保存可视化图片的目录
    visual_dir = os.path.join(culane_root, CULaneDIR.images_dir + "-visual")
    # if os.path.exists(visual_dir) and IS_RM_LEAGACY:
    #     shutil.rmtree(visual_dir)
    os.makedirs(visual_dir, exist_ok=True)

    return images_dir, labels_dir, list_dir, visual_dir


def convert_label(json_file: str):
    def check_key(json_data: dict, key: str):
        if key not in json_data:
            raise KeyError(f"Not found key:'{key}' in {json_file}")

    with open(json_file, 'r') as f:
        json_data = json.load(f)
        check_key(json_data, "shapes")
        shapes_data = json_data["shapes"]
        lanes = {str(id): [] for id in LANE_DEFINE}
        for shape in shapes_data:
            check_key(shape, "label")
            check_key(shape, "points")
            lanes[shape["label"]] = shape["points"]
    return lanes


if __name__ == "__main__":
    main()