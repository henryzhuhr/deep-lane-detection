import os
import time
import cv2
import numpy as np
from utils.onnx_infer import ONNXInfer, InferResult


class Args:
    weight_file = "../weights/train/ufld-final.onnx"
    # 侧方
    image_file = "/Users/henryzhu/datasets/CULane-custom/images/230721_144432-0020.jpg"
    # 倒车
    # image_file = "/Users/henryzhu/datasets/CULane-custom/images/230721_150133-0002.jpg"


def main():
    model_infer = ONNXInfer(Args.weight_file,0.5)

    frame = cv2.imread(Args.image_file)

    st = time.time()

    img_w = 800    # 缩放图像
    img_h = frame.shape[0] * img_w // frame.shape[1]

    # 裁切下半部分，裁切高度为 288 倍数 或者 0
    crop_h = int(img_w * 0.36) # 0.36 = 288 / 800
                               # crop_h = 0                 # 如果不裁切，可以改成 0

    img = cv2.resize(frame, (img_w, img_h))
    # img=cv2.flip(img,2,dst=None) # 水平翻转

    add_h = img_h - crop_h
    crop_img = img[add_h : img_h, :, :] # 裁切下半部分

    infer_result: InferResult = model_infer.infer(crop_img, False)

    # 裁切后 y 坐标需要加上偏移量
    for i in range(infer_result.lanes_y_coords.shape[0]):
        infer_result.lanes_y_coords[i] += add_h
    for i in range(infer_result.forward_direct.shape[0]):
        infer_result.forward_direct[i][1] += add_h
    for i in range(infer_result.predict_direct.shape[0]):
        infer_result.predict_direct[i][1] += add_h

    lane_y_coords = infer_result.lanes_y_coords              # [18]     所有车道线共用一组 y 坐标
    lanes_x_coords = infer_result.lanes_x_coords             # [4, 18]  4 个车道线的 x 坐标
    lanes_x_coords_kl = infer_result.lanes_x_coords_kl       # [4, 18]  卡尔曼滤波后 4 个车道线的 x 坐标
    lane_center_x_coords = infer_result.lane_center_x_coords # [18]     中心车道线的 x 坐标
    forwarddirect = infer_result.forward_direct              # [2, 2]   实际前进方向
    predict_direct = infer_result.predict_direct             # [2, 2]   预测前进方向
    y_offset = infer_result.y_offset
    z_offset = infer_result.z_offset

    if True:                                            # 如果不需要绘制，可以改成 False
        cv2.rectangle(img, (0, add_h), (img_w, img_h), (0, 255, 0), 2)
        img = model_infer.mark_result(img, infer_result)
        
    if False:
        right_lane = [[int(x), int(y)] for x, y in zip(lanes_x_coords[2], lane_y_coords) if x != 0]
        if len(right_lane) > 0:
            top_point = right_lane[len(right_lane) - 1] # 右车道线最上面的的点
            sub_img = img.copy()[top_point[1]: img_h, top_point[0]: img_w]

            # 二值化，增强对比度
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
            sub_img = cv2.GaussianBlur(sub_img, (3, 3), 0)
            sub_img = cv2.equalizeHist(sub_img)
            sub_img = cv2.threshold(sub_img, 240, 255, cv2.THRESH_BINARY)[1] # [240，255]

            # 创建一个 3 通道黑白图
            sub_img_show = cv2.cvtColor(sub_img.copy(), cv2.COLOR_GRAY2BGR)

            canny_img = cv2.Canny(sub_img, 50, 150)
            hough_lines = cv2.HoughLinesP(
                canny_img,
                2,                         # rho: 线段以像素为单位的距离精度，double类型的，推荐用1.0
                np.pi / 180,               # theta： 线段以弧度为单位的角度精度，推荐用numpy.pi/180
                40,                        # threshod: 累加平面的阈值参数，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。
                minLineLength=20,          # 线段以像素为单位的最小长度，根据应用场景设置
                maxLineGap=20,             # 同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
            )

            detected_lines = [line_[0] for line_ in hough_lines]
            # 直线方程 y = k * x + b
            lane_left_eq = np.polyfit(
                np.array([x for x, y in right_lane], np.int32),
                np.array([y for x, y in right_lane], np.int32),
                1,
            )

            linear_equations = []
            for x1, y1, x2, y2 in detected_lines:
                k = (y2 - y1) / (x2 - x1)
                b = y1 - k * x1
                slope_row = 0.3 # 横线大概斜率
                                # 只考虑加入横线
                if (k > 0 and k < slope_row) or (k < 0 and k > -1 * slope_row):
                    is_append = True
                                # 去除平行且相近的线
                    for j in range(len(linear_equations)):
                        kj, bj, _ = linear_equations[j]
                        if abs(k - kj) < 0.1 and abs(b - bj) < 5:
                            is_append = False
                            break
                    if is_append:
                        linear_equations.append([k, b, [x1, y1, x2, y2]])
                        color = np.random.randint(0, 255, (3)).tolist()
                        cv2.line(sub_img_show, (x1, y1), (x2, y2), color, 2)

            # 求解交点
            intersect_pts = []
            for i in range(len(linear_equations)):
                k1, b1, _ = linear_equations[i]
                k2, b2 = lane_left_eq
                if k1 == k2:
                    continue
                x = (b2 - b1) / (k1 - k2)
                y = k1 * x + b1
                is_append = True
                pix_thred = img_h * 0.1
                for _, x_, y_ in intersect_pts:
                    # 交点去重
                    if abs(x - x_) < pix_thred or abs(y - y_) < pix_thred:
                        is_append = False
                        break
                if is_append:
                    intersect_pts.append([i, x, y])
            for i, x, y in intersect_pts:
                _, _, line_p = linear_equations[i]
                x_1, y_1, x_2, y_2 = line_p

                cv2.circle(sub_img_show, (int(x), int(y)), 10, (0, 0, 255), -1)
                cv2.line(sub_img_show, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)

                cv2.circle(img, (int(x + top_point[0]), int(y + top_point[1])), 10, (0, 0, 255), -1)
                cv2.line(
                    img, (x_1 + top_point[0], y_1 + top_point[1]), (x_2 + top_point[0], y_2 + top_point[1]),
                    (0, 0, 255), 2
                )

            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(sub_img)[0]
            lines = np.squeeze(lines, axis=1).astype(np.int32)
            for i in range(lines.shape[0]):
                p0 = lines[i][0 : 2]
                p1 = lines[i][2 : 4]
                # cv2.line(sub_img_show, p0, p1, (0, 0, 255), 1)

            for i in range(len(right_lane)):
                right_lane[i][0] -= top_point[0]
                right_lane[i][1] -= top_point[1]

            # for i in range(len(right_lane) - 1):
            #     p0 = right_lane[i]
            #     p1 = right_lane[i + 1]
            #     cv2.circle(sub_img_show, p0, 3, (0, 255, 0), -1)
            #     cv2.line(sub_img_show, p0, p1, (0, 255, 0), 1)
            # cv2.circle(sub_img_show, top_point, 5, (0, 0, 255), -1)

    # 推理时间
    infer_time = (time.time() - st) * 1000
    print("time: %.4f ms" % infer_time)

    os.makedirs("tmps", exist_ok=True)
    cv2.imwrite("tmps/infer-onnx.jpg", img)
    # cv2.imwrite("tmps/infer-onnx-sub.jpg", sub_img_show)


if __name__ == "__main__":
    main()