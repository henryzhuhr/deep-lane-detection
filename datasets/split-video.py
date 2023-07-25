import base64
import datetime
import json
import os
import time
import cv2
import tqdm

from common import get_culane_root
from lane_detector import LaneDetector

SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]


def main():

    culane_root = get_culane_root()

    videos_dir = os.path.join(culane_root, "videos")
    images_dir = os.path.join(culane_root, "images")
    os.makedirs(images_dir, exist_ok=True)

    for video_file in os.listdir(videos_dir):
        if os.path.splitext(video_file)[1] in SUPPORTED_VIDEO_FORMATS:
            video_path = os.path.join(videos_dir, video_file)
            split_video(video_path, images_dir)


def split_video(video_path: str, images_dir: str):
    video_file_name = os.path.splitext(os.path.split(video_path)[1])[0]

    video_image_dir = os.path.join(images_dir, video_file_name)

    if os.path.exists(video_image_dir):
        raise FileExistsError(
            f"\033[01;33mfound '{video_image_dir}', If you need to regenerate it, please delete the directory.\033[0m"
        )

    os.makedirs(video_image_dir)

    cap = cv2.VideoCapture(video_path)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lane_detector = LaneDetector(img_size=(height, width))

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
            pbar.set_description(f"index:{sec_idx:04d}")
            save_name = f"{video_file_name}-{sec_idx:04d}"
                                   # save image
            cv2.imwrite(os.path.join(video_image_dir, save_name + ".jpg"), frame)


if __name__ == '__main__':
    main()