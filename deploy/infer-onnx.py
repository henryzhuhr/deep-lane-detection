import os
import time

import cv2
import numpy as np
from utils.onnx_infer import ONNXInfer, InferResult


class Args:
    weight_file = "../weights/ufld-final.onnx"

    video = "../../examples/lane-detection-demo.mp4"
    video = os.path.expandvars("$HOME/Downloads/IMG_5281.MOV") # 视频文件


def main():
    model_infer = ONNXInfer(Args.weight_file)

    cap = cv2.VideoCapture(Args.video)
    skip_frame = 1 # 每隔 skip_frame 帧进行一次推理
    i = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            i += 1
            if i % skip_frame != 0:
                continue
                # i = 0

            img = frame
            st = time.time()

            img_w = 800 # 缩放图像
            img_h = frame.shape[0] * img_w // frame.shape[1]

            # 裁切下半部分，裁切高度为 288 倍数 或者 0
            crop_h = int(img_w * 0.36) # 0.36 = 288 / 800
            # crop_h = 0                 # 如果不裁切，可以改成 0

            img = cv2.resize(frame, (img_w, img_h))

            start_h = img_h - crop_h
            crop_img = img[start_h : img_h, :, :] # 裁切下半部分

            infer_result: InferResult = model_infer.infer(crop_img)

            for i in range(infer_result.lanes_y_coords.shape[0]):
                infer_result.lanes_y_coords[i] += start_h
            for i in range(infer_result.forward_direct.shape[0]):
                infer_result.forward_direct[i][1] += start_h
            for i in range(infer_result.predict_direct.shape[0]):
                infer_result.predict_direct[i][1] += start_h

            lane_y_coords = infer_result.lanes_y_coords              # [18]     所有车道线共用一组 y 坐标
            lanes_x_coords = infer_result.lanes_x_coords             # [4, 18]  4 个车道线的 x 坐标
            lanes_x_coords_kl = infer_result.lanes_x_coords_kl       # [4, 18]  卡尔曼滤波后 4 个车道线的 x 坐标
            lane_center_x_coords = infer_result.lane_center_x_coords # [18]     中心车道线的 x 坐标
            forward_direct = infer_result.forward_direct             # [2, 2]   实际前进方向
            predict_direct = infer_result.predict_direct             # [2, 2]   预测前进方向
            slope = infer_result.slope                               # 预测方向的斜率
            offset_distance = infer_result.offset_distance           # 偏移距离

            if True: # 如果不需要绘制，可以改成 False
                img=cv2.rectangle(img, (0, start_h), (img_w, img_h), (0, 255, 0), 2)
                img = model_infer.mark_result(img, infer_result)

            # 推理时间
            infer_time = (time.time() - st) * 1000
            print("time: %.4f ms" % infer_time)

            cv2.imshow("img", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()