import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image_file = "/Users/henryzhu/datasets/CULane-custom/images/230721_150133-0002.jpg"
 
image = cv2.imread(image_file,0)
 
print('This image is:', type(image), 'with dimensions:', image.shape)
import math
 
def grayscale(img):
    """
    将图像处理为灰度图像，因为使用cv2read所以要用BGR进行转换
    
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):#返回image，边缘部分为255，其余为0
 
    return cv2.Canny(img, low_threshold, high_threshold)
 
def gaussian_blur(img, kernel_size):
 
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
 
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    print("mask_shape",mask.shape)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
 
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
 
def draw_lines(img, lines, color=[255,116,10], thickness=2):
        left_lines_x = []
        left_lines_y = []
        right_lines_x = []
        right_lines_y = []
        line_y_max = 0
        line_y_min = 999
        for line in lines:
            for x1,y1,x2,y2 in line:
                if y1 > line_y_max:
                    line_y_max = y1
                if y2 > line_y_max:
                    line_y_max = y2
                if y1 < line_y_min:
                    line_y_min = y1
                if y2 < line_y_min:
                    line_y_min = y2
                k = (y2 - y1)/(x2 - x1)
                if k < -0.3:
                    left_lines_x.append(x1)
                    left_lines_y.append(y1)
                    left_lines_x.append(x2)
                    left_lines_y.append(y2)
                elif k > 0.3:
 
                    right_lines_x.append(x1)
                    right_lines_y.append(y1)
                    right_lines_x.append(x2)
                    right_lines_y.append(y2)
        #最小二乘直线拟合
        left_line_k, left_line_b = np.polyfit(left_lines_x, left_lines_y, 1)
        right_line_k, right_line_b = np.polyfit(right_lines_x, right_lines_y, 1)
        #根据直线方程和最大、最小的y值反算对应的x
        cv2.line(img,(int((line_y_max - left_line_b)/left_line_k), line_y_max),(int((line_y_min - left_line_b)/left_line_k), line_y_min),color, thickness)
        cv2.line(img,(int((line_y_max - right_line_b)/right_line_k), line_y_max),(int((line_y_min - right_line_b)/right_line_k), line_y_min),color, thickness)
 
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    print(lines.shape)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
 
 
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
 
    return cv2.addWeighted(initial_img, α, img, β, γ)
 
ini_image = cv2.imread(image_file)
#plt.imshow(ini_image)
imshape=ini_image.shape
gray=grayscale(ini_image)
#after灰度处理
 
kernel_size=5
blur=gaussian_blur(gray, kernel_size)
 
#after高斯模糊
low_threshold=50
high_threshold=200
edges=canny(blur, low_threshold, high_threshold)
 
vertices=np.array([[(0,imshape[0]),(imshape[1]/2-20, imshape[0]/2+50),
                        (imshape[1]/2+20, imshape[0]/2+50), 
                    (imshape[1],imshape[0]),(0,500),(960,500)]],
                      dtype=np.int32)
 
partial=region_of_interest(edges,vertices)
x=[0,0,460,500,960,960]
y=[540,500,320,320,500,540]
plt.plot(x,y)
plt.imshow(edges)
 
rho=1
theta=np.pi/180
threshold=13
min_line_len=15
max_line_gap=10
lines=hough_lines(partial, rho, theta, threshold, min_line_len, max_line_gap)
 
final=weighted_img(lines,ini_image)
ini_image = cv2.imread(image_file)
imshape=ini_image.shape
gray=grayscale(ini_image)
#after灰度处理
 
kernel_size=5
blur=gaussian_blur(gray, kernel_size)
#after高斯模糊
 
 
low_threshold=120
high_threshold=200
edges=canny(blur, low_threshold, high_threshold)
#aftercannoy滤波
vertices=np.array([[(0,imshape[0]),(imshape[1]/2-20, imshape[0]/2+50),  \
                        (imshape[1]/2+20, imshape[0]/2+50), (imshape[1],imshape[0]),(0,500),(960,500)]], \
                      dtype=np.int32)
 
 
partial=region_of_interest(edges,vertices)
 
rho=1
theta=np.pi/180
threshold=13
min_line_len=15
max_line_gap=10
lines=hough_lines(partial, rho, theta, threshold, min_line_len, max_line_gap)
final=weighted_img(lines,ini_image)
cv2.imshow('Final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
