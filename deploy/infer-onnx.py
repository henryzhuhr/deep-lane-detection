import os
import time
import cv2
import numpy as np
from utils.onnx_infer import ONNXInfer, InferResult


def color_extraction(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # range of yellow and white
    yellow_lower = np.array([20, 125, 125])  # yellow
    yellow_upper = np.array([35, 255, 255])
    white_lower = np.array([0, 0, 200])  # white
    white_upper = np.array([180, 30, 255])

    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

    processed_image = cv2.bitwise_and(image, image, mask=combined_mask)

    return processed_image

def undistort_image(image, camera_matrix, distortion_coefficients):
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients)
    return undistorted_image


class Args:
    weight_file = "../weights/culane_18.onnx"

    video = "../examples/IMG_5281.MOV"


def main():
    model_infer = ONNXInfer(Args.weight_file)

    camera_matrix = np.array([[654.70620869, 0.00000000e+00, 620.18116133], [0.00000000e+00, 656.66098208, 433.29648979], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    distortion_coefficients = np.array([-0.32841624,  0.10118464, 0.00383333,  0.00071582, -0.01392005])

    # cap = cv2.VideoCapture(Args.video)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    skip_frame = 2  # 每隔 skip_frame 帧进行一次推理
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
                                       # crop_h = 0  # 如果不裁切，可以改成 0

            img=img[200:, :]
            # img = cv2.resize(frame, (img_w, img_h))

            

            add_h = img_h - 0
            crop_img = img[add_h : img_h, :, :] # 裁切下半部分
            img = undistort_image(img, camera_matrix, distortion_coefficients)
            img = img[:, :1200, :]  # 裁切下半部分
            # img = color_extraction(img)

            infer_result: InferResult = model_infer.infer(img)

            # 裁切后 y 坐标需要加上偏移量
            # for i in range(infer_result.lanes_y_coords.shape[0]):
            #     infer_result.lanes_y_coords[i] += add_h
            # for i in range(infer_result.forward_direct.shape[0]):
            #     infer_result.forward_direct[i][1] += add_h
            # for i in range(infer_result.predict_direct.shape[0]):
            #     infer_result.predict_direct[i][1] += add_h

            lane_y_coords = infer_result.lanes_y_coords              # [18]     所有车道线共用一组 y 坐标
            lanes_x_coords = infer_result.lanes_x_coords             # [4, 18]  4 个车道线的 x 坐标
            lanes_x_coords_kl = infer_result.lanes_x_coords_kl       # [4, 18]  卡尔曼滤波后 4 个车道线的 x 坐标
            lane_center_x_coords = infer_result.lane_center_x_coords # [18]     中心车道线的 x 坐标
            forward_direct = infer_result.forward_direct             # [2, 2]   实际前进方向
            predict_direct = infer_result.predict_direct             # [2, 2]   预测前进方向
            y_offset = infer_result.y_offset
            z_offset = infer_result.z_offset

            if True: # 如果不需要绘制，可以改成 False
                # cv2.rectangle(img, (0, add_h), (img_w, img_h), (0, 255, 0), 2)
                img = model_infer.mark_result(img, infer_result)

            # for i in range(lane_y_coords.shape):
            #     x =

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