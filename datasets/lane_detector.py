import numpy as np
import cv2


class LaneDetector:
    def __init__(self, img_size=[720, 1280]) -> None:
        self.left__line = LaneLine() # 记录检测到的车道线
        self.right_line = LaneLine()
        self.img_size = img_size     # H, W

    def get_line_points(self, img: cv2.Mat):
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img_h, img_w = img.shape[: 2]

        (fitered_img, _, _) = filter_colors(img)
        grayscale = cv2.cvtColor(fitered_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
        (T, thres) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        (warped, unwarped, perspective_transform_matrix, perspective_transform_invmatrix) = warp_perspective(thres)

        # Function to detect lanes
        lane_detected, cur = readjust_line_search(warped, self.left__line, self.right_line)

        # img=cv2.cvtColor(thres,cv2.COLOR_GRAY2BGR)

        img, left_line_points = sample_points(
            img, self.left__line.all_of_x, self.left__line.all_of_y, perspective_transform_invmatrix
        )
        img, right_line_points = sample_points(
            img, self.right_line.all_of_x, self.right_line.all_of_y, perspective_transform_invmatrix
        )
        return img,{
            "1": left_line_points,
            "2": right_line_points,
        }


def sample_points(img, points_x, points_y, trans_mat):
    sample_point_num = 10 # 采样点数量
    line_points = [cvt_pos(x, y, trans_mat) for [x, y] in zip(points_x, points_y)]
    new_y_points = [y for [x, y] in line_points]

    y_list = np.linspace(np.min(new_y_points), np.max(new_y_points), sample_point_num)
    sample_line_points = []
    # 找到和 y_list 最临近的点
    for y in y_list:
        min_dist = 9999
        min_point = None
        for point in line_points:
            dist = np.abs(y - point[1])
            if dist < min_dist:
                min_dist = dist
                min_point = point
        if min_point is not None:
            sample_line_points.append(min_point)
            cv2.circle(img, [int(p) for p in min_point], 5, (0, 0, 255), -1)
    return img, sample_line_points


def cvt_pos(u, v, mat):
    x = (mat[0][0] * u + mat[0][1] * v + mat[0][2]) / (mat[2][0] * u + mat[2][1] * v + mat[2][2])
    y = (mat[1][0] * u + mat[1][1] * v + mat[1][2]) / (mat[2][0] * u + mat[2][1] * v + mat[2][2])
    return [x,y]


class LaneLine:
    """
    车道线，存储检测结果
    """
    def __init__(self):
        self.found = False                     # Lane line found in the previous iteration
        self.window_size_limits = 56           # Window width with limits
        self.previous_x = []                   # X values of last iterations
        self.present_fit = [np.array([False])] # Coefficients of recent fit polynomial
        self.r_o_c = None                      # Radius of curvature of lane line
        self.x_start = None                    # Starting x value
        self.x_end = None                      # Ending x value
        self.all_of_x = None                   # Values of x for found lane line
        self.all_of_y = None                   # Values of y for found lane line
        self.information_of_road = None        # Metadata
        self.curvature = None
        self.divergence = None


def filter_colors(img: cv2.Mat):
    """
    过滤黄色和白色车道线，返回过滤后的图像
    """
    # Filter white pixels
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(img, lower_white, upper_white)
    white_img = cv2.bitwise_and(img, img, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([0, 80, 80])
    upper_yellow = np.array([110, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_img = cv2.bitwise_and(img, img, mask=yellow_mask)

    # Combine the two above imgs
    fitered_img = cv2.addWeighted(white_img, 1., yellow_img, 1., 0.)

    return (
        fitered_img, # 叠加图像
        white_img,   # 白色车道线图像
        yellow_img,  # 黄色车道线图像
    )


def warp_perspective(img: cv2.Mat):
    """
        透视变换，需要标定
    """

    # warp the image
    img_h, img_w = img.shape
    src = np.float32(
        [
            [200 / 1280 * img_w, 720 / 720 * img_h], [1100 / 1280 * img_w, 720 / 720 * img_h],
            [595 / 1280 * img_w, 450 / 720 * img_h], [685 / 1280 * img_w, 450 / 720 * img_h]
        ]
    )
    dst = np.float32(
        [[300 / 1280 * img_w, img_h], [980 / 1280 * img_w, img_h], [300 / 1280 * img_w, 0], [980 / 1280 * img_w, 0]]
    )

    perspective_transform_matrix = cv2.getPerspectiveTransform(src, dst) # transformation matrix
    perspective_transform_invmatrix = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(
        img,
        perspective_transform_matrix,
        (img_w, img_h),
        flags=cv2.INTER_LINEAR,
    )
    unwarped = cv2.warpPerspective(
        warped,
        perspective_transform_invmatrix,
        (warped.shape[1], warped.shape[0]),
        flags=cv2.INTER_LINEAR,
    )                                       # DEBUG
    return (
        warped,
        unwarped,
        perspective_transform_matrix,       # 透视变换矩阵
        perspective_transform_invmatrix,    # 透视变换逆矩阵
    )



def readjust_line_search(img, left_lane, right_lane):
    # 图像下半部分沿列的直方图 Histogram in lower half of image along columns
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)

    # Blank canvas
    res_img = np.dstack((img, img, img)) * 255

    # Find peak values of left and right halves of histogram
    middle = np.int32(histogram.shape[0] / 2)
    left_p = np.argmax(histogram[: middle])
    right_p = np.argmax(histogram[middle :]) + middle

    # Number of sliding windows
    window_number = 9
    # Define window height
    height_of_window = np.int32(img.shape[0] / window_number)
    # Find all non zero pixel
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    # Present positions to be updated for each window
    present_left_x = left_p
    present_right_x = right_p
    # Min number of pixels found to recenter window
    min_number_pixel = 50
    # Empty lists to save left and right lane pixel indices
    window_left_lane = []
    window_right_lane = []
    window_margin = left_lane.window_size_limits
    # Go through the windows one after other
    for window in range(window_number):
        # Find window boundaries
        win_y_low = img.shape[0] - (window + 1) * height_of_window
        win_y_high = img.shape[0] - window * height_of_window
        win_leftx_min = present_left_x - window_margin
        win_leftx_max = present_left_x + window_margin
        win_rightx_min = present_right_x - window_margin
        win_rightx_max = present_right_x + window_margin
        # Draw windows on canvas
        cv2.rectangle(res_img, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(res_img, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)
        # Find nonzero pixels inside window
        left_window_inds = (
            (nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_leftx_min) &
            (nonzero_x <= win_leftx_max)
        ).nonzero()[0]
        right_window_inds = (
            (nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_rightx_min) &
            (nonzero_x <= win_rightx_max)
        ).nonzero()[0]
        # Append indices to list
        window_left_lane.append(left_window_inds)
        window_right_lane.append(right_window_inds)
        # If found > minpixels, recenter next window to their mean position
        if len(left_window_inds) > min_number_pixel:
            present_left_x = np.int32(np.mean(nonzero_x[left_window_inds]))
        if len(right_window_inds) > min_number_pixel:
            present_right_x = np.int32(np.mean(nonzero_x[right_window_inds]))

    # Concatenate the arrays of indoices
    window_left_lane = np.concatenate(window_left_lane)
    window_right_lane = np.concatenate(window_right_lane)
    # Extract left and right line pixel positions
    leftx = nonzero_x[window_left_lane]
    lefty = nonzero_y[window_left_lane]
    rightx = nonzero_x[window_right_lane]
    righty = nonzero_y[window_right_lane]
    res_img[lefty, leftx] = [255, 0, 0]
    res_img[righty, rightx] = [0, 0, 255]
    # Fit second order polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    # print(righty)
    right_fit = np.polyfit(righty, rightx, 2)
    left_lane.present_fit = left_fit
    right_lane.present_fit = right_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_plotx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    left_lane.previous_x.append(left_plotx)
    right_lane.previous_x.append(right_plotx)
    if len(left_lane.previous_x) > 10:
        left_avg_line = even_off(left_lane.previous_x, 10,img_h=img.shape[0])
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty**2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_lane.present_fit = left_avg_fit
        left_lane.all_of_x, left_lane.all_of_y = left_fit_plotx, ploty
    else:
        left_lane.present_fit = left_fit
        left_lane.all_of_x, left_lane.all_of_y = left_plotx, ploty
    if len(right_lane.previous_x) > 10:
        right_avg_line = even_off(right_lane.previous_x, 10,img_h=img.shape[0])
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty**2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_lane.present_fit = right_avg_fit
        right_lane.all_of_x, right_lane.all_of_y = right_fit_plotx, ploty
    else:
        right_lane.present_fit = right_fit
        right_lane.all_of_x, right_lane.all_of_y = right_plotx, ploty

    left_lane.x_start, right_lane.x_start = left_lane.all_of_x[len(left_lane.all_of_x) -
                                                               1], right_lane.all_of_x[len(right_lane.all_of_x) - 1]
    left_lane.x_end, right_lane.x_end = left_lane.all_of_x[0], right_lane.all_of_x[0]
    left_lane.found, right_lane.found = True, True
    cur = compute_curvature(left_lane, right_lane, size=img.shape)
    return res_img, cur

def even_off(lines, number_of_previous_lines=3, img_h=720):
    # Average out lines
    lines = np.squeeze(lines)
    averaged_line = np.zeros((img_h))
    for i, line in enumerate(reversed(lines)):
        if i == number_of_previous_lines:
            break
        averaged_line += line
    averaged_line = averaged_line / number_of_previous_lines
    return averaged_line


def compute_curvature(left_lane_line, right_lane_line, size):
    y = left_lane_line.all_of_y
    left_x, right_x = left_lane_line.all_of_x, right_lane_line.all_of_x
    # Flip to match openCV Y direction
    left_x = left_x[::-1]
    right_x = right_x[::-1]
    # Max value of y -> bottom
    y_value = np.max(y)
    # Calculation and conversion roc meter per pixel
    lane_width = abs(right_lane_line.x_start - left_lane_line.x_start)
    y_m_per_pixel = 30 / size[0]                           # 30 / 720
    x_m_per_pixel = 3.7 * (size[0] / size[1]) / lane_width # 3.7 * (720 / 1280) / lane_width

    # Fit polynomial in world space
    left_curve_fit = np.polyfit(y * y_m_per_pixel, left_x * x_m_per_pixel, 2)
    right_curve_fit = np.polyfit(y * y_m_per_pixel, right_x * x_m_per_pixel, 2)

    # Compute new roc
    # TODO: RuntimeWarning: invalid value encountered in power
    # np.poewr (x,1.5) x< 0 的情况 print(left_curve_fit,right_curve_fit)
    left_curve_fit_radius = (
        np.power((1 + np.power((2 * left_curve_fit[0] * y_value * y_m_per_pixel + left_curve_fit[1]), 2)), 1.5)
    ) / np.absolute(2 * left_curve_fit[0])
    right_curve_fit_radius = (
        (1 + np.power((2 * right_curve_fit[0] * y_value * y_m_per_pixel + np.power(right_curve_fit[1], 2)), 1.5)) /
        np.absolute(2 * right_curve_fit[0])
    )

    # ROC
    left_lane_line.r_o_c = left_curve_fit_radius
    right_lane_line.r_o_c = right_curve_fit_radius
    return left_curve_fit_radius
