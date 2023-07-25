import base64
import datetime
import json
import os
import sys
import time
import cv2
import tqdm
from common import get_culane_root

current_path = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_path)
print(parent_dir)
sys.path.append(os.path.join(parent_dir, 'deploy'))
from utils.onnx_infer import ONNXInfer, InferResult

SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]


def main():

    model_infer = ONNXInfer("weights/ufld-final.onnx")
    culane_root = get_culane_root()
    culane_root = '/Users/henryzhu/datasets/CULane-custom'

    videos_dir = os.path.join(culane_root, "videos")
    images_dir = os.path.join(culane_root, "videos-split")
    os.makedirs(images_dir, exist_ok=True)

    for video_file in os.listdir(videos_dir):
        if os.path.splitext(video_file)[1] in SUPPORTED_VIDEO_FORMATS:
            video_path = os.path.join(videos_dir, video_file)
            split_video(model_infer, video_path, images_dir)


def split_video(model_infer: ONNXInfer, video_path: str, images_dir: str):
    video_file_name = os.path.splitext(os.path.split(video_path)[1])[0]

    video_image_dir = os.path.join(images_dir, video_file_name)

    if os.path.exists(video_image_dir):
        raise FileExistsError(
            f"\033[01;33mfound '{video_image_dir}', If you need to regenerate it, please delete the directory.\033[0m"
        )

    os.makedirs(video_image_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ctime = os.path.getctime(video_path) #创建时间
    create_date = time.strftime("%Y%m%d_%H%M%S", time.localtime(ctime))

    print(f"Process:{video_path}")
    print(f"fps:{fps}", f"size:[{width},{height}]", f"frame_counter:{frame_counter}", f"create_date:{create_date}")

    pbar = tqdm.tqdm(total=frame_counter)

    skip = int(fps / int(fps / 2)) # 每0.5秒读取一次图像，避免图像过于相似
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        i += 1
        if i % skip == 0:
            sec_idx = i // skip
            pbar.set_description(f"[{i}] index:{sec_idx:04d}")

            save_name = f"{video_file_name}-{sec_idx:04d}"
            infer_result: InferResult = model_infer.infer(frame.copy(), is_kalman=False)

            lane_y_coords = infer_result.lanes_y_coords  # [18]     所有车道线共用一组 y 坐标
            lanes_x_coords = infer_result.lanes_x_coords # [4, 18]  4 个车道线的 x 坐标

            shapes = []
            for label_id in range(4):
                points = []
                for x, y in zip(lanes_x_coords[label_id].tolist(), lane_y_coords.tolist()):
                    if x != 0 and y != 0:
                        points.append([x, y])
                if len(points) > 0:
                    shapes.append(
                        {
                            "label": f"{label_id+1}",
                            "points": points,
                            "group_id": None,
                            "description": "",
                            "shape_type": "linestrip",
                            "flags": {}
                        }
                    )

            # save image
            cv2.imwrite(os.path.join(video_image_dir, save_name + ".jpg"), frame)

            # save label as .json 对标签进行预先标注，转化为 labelme 格式
            with open(os.path.join(video_image_dir, save_name + ".json"), "w") as f:
                json.dump(
                    {
                        "version": "5.2.1",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": save_name + ".jpg",
                        "imageData": str(base64.b64encode(cv2.imencode('.jpg', frame)[1]))[2 :-1],
                        "imageHeight": height,
                        "imageWidth": width
                    },
                    f,
                    indent=4
                )


if __name__ == '__main__':
    main()