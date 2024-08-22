import cv2
import numpy as np

cap = cv2.VideoCapture('driving.mp4')
output = None

# get h and w
ret, frame = cap.read()
h, w, _ = frame.shape
x_mid = w // 2
y_mid = h // 2

# visualizations
lane_color = (0, 255, 255)
lane_fill_color = (0, 255, 0)
lane_fill_opacity = 0.2

# parameters
gaussian_kernel_size = 9

canny_min = 60
canny_max = 100

rho = 1
theta = np.pi/180
threshold = 50
min_line_len = w // 4
max_line_gap = w

min_lane_len = w // 10
# the farthest right the left lane goes
left_lane_x_center = int(0.45 * w)
right_lane_x_center = int(0.55 * w)

# prev variables
prev_left_lane_x_intercept = 0
prev_left_lane_at_center = y_mid

prev_right_lane_x_intercept = w
prev_right_lane_at_center = y_mid

# remove data more than n std dev from the mean
def remove_outliers(data, n=2):
    data = np.array(data)
    mean = np.mean(data)
    stddev = max(len(data), np.std(data))
    
    result = data[abs(data - mean) < n * stddev]
    if len(result) == 0:
        return np.array([mean])
    return result

while cap.isOpened():
    # get frames
    ret, frame = cap.read()
    if not ret:
        break
    print("new frame")
        
    # set up video writer
    h, w, _ = frame.shape
    if not output:
        output = cv2.VideoWriter("detected_lanes.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h), True)
        
    
    # grayscale and filter to get edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.GaussianBlur(gray, (gaussian_kernel_size, gaussian_kernel_size), 0)
    edges = cv2.Canny(filtered, canny_min, canny_max)
    
    
    # outline mask with vertices and fill the poly
    mask = np.zeros_like(edges)
    vertices = np.array([[(0,h), (x_mid,y_mid), (w,h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    
    # apply mask
    masked = cv2.bitwise_and(edges, mask)
    
    
    # detect straight lines with hough lines
    lines = cv2.HoughLinesP(masked, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    
    # draw lines
#    if lines is not None:
#        for line in lines:
#            l = line[0]
#            x1, y1, x2, y2 = l
#            cv2.line(masked, (x1, y1), (x2, y2), (255,0,0), 5)
    
    left_lane_x_values = []
    left_lane_y_values = []
    left_lane_slopes = []
    right_lane_x_values = []
    right_lane_y_values = []
    right_lane_slopes = []

    if lines is not None:
        for line in lines:
            l = line[0]
            # y in images is opposite of a coordinate plane
            x1, y1, x2, y2 = l[0], h-l[1], l[2], h-l[3]

            # if the line is long enough
            if ((x2 - x1) ** 2 + (y2 - y1) ** 2) > min_lane_len ** 2:
                if x1 > x_mid and x2 > x_mid and x1 != x2: # right lane
                    right_lane_x_values.append((x1 + x2) / 2)
                    right_lane_y_values.append((y1 + y2) / 2)
                    right_lane_slopes.append((y2 - y1) / (x2 - x1))
                elif x1 < x_mid and x2 < x_mid and x1 != x2: # left lane
                    left_lane_x_values.append((x1 + x2) / 2)
                    left_lane_y_values.append((y1 + y2) / 2)
                    left_lane_slopes.append((y2 - y1) / (x2 - x1))
                    
    lane_boundaries = []
                    
    # draw left lane if available, otherwise use old left lane
    if len(left_lane_x_values) > 0 and len(left_lane_y_values) > 0 and len(left_lane_slopes) > 0:
        left_lane_x = np.mean(remove_outliers(left_lane_x_values))
        left_lane_y = np.mean(remove_outliers(left_lane_y_values))
        left_lane_slope = np.mean(remove_outliers(left_lane_slopes))
        
        # draw line using x intercept and its value at x_mid
        left_lane_x_intercept = int(-left_lane_y / left_lane_slope + left_lane_x)
        left_lane_at_center = int((left_lane_x_center - left_lane_x) * left_lane_slope + left_lane_y)
        
        # save prev values
        prev_left_lane_x_intercept = left_lane_x_intercept
        prev_left_lane_at_center = left_lane_at_center
        
        # y in images is opposite of a coordinate plane
        cv2.line(frame, (left_lane_x_intercept,h), (left_lane_x_center, int(h - left_lane_at_center)), lane_color, 5)
        
        lane_boundaries.append((left_lane_x_intercept, h))
        lane_boundaries.append((left_lane_x_center, int(h - left_lane_at_center)))
    else:
        cv2.line(frame, (prev_left_lane_x_intercept,h), (left_lane_x_center, int(h - prev_left_lane_at_center)), lane_color, 5)
        lane_boundaries.append((prev_left_lane_x_intercept,h))
        lane_boundaries.append((left_lane_x_center, int(h - prev_left_lane_at_center)))

    # draw right lane if available, otherwise use old right lane
    if len(right_lane_x_values) > 0 and len(right_lane_y_values) > 0 and len(right_lane_slopes) > 0:
        right_lane_x = np.mean(remove_outliers(right_lane_x_values))
        right_lane_y = np.mean(remove_outliers(right_lane_y_values))
        right_lane_slope = np.mean(remove_outliers(right_lane_slopes))
        
        # draw line using x intercept and its value at x_mid
        right_lane_x_intercept = int(-right_lane_y / right_lane_slope + right_lane_x)
        right_lane_at_center = int((right_lane_x_center - right_lane_x) * right_lane_slope + right_lane_y)
        
        # save prev values
        prev_right_lane_x_intercept = right_lane_x_intercept
        prev_right_lane_at_center = right_lane_at_center

        # y in images is opposite of a coordinate plane
        cv2.line(frame, (right_lane_x_intercept,h), (right_lane_x_center, int(h - right_lane_at_center)), lane_color, 5)
        
        lane_boundaries.append((right_lane_x_center, int(h - right_lane_at_center)))
        lane_boundaries.append((right_lane_x_intercept, h))
    else:
        cv2.line(frame, (prev_right_lane_x_intercept,h), (right_lane_x_center, int(h - prev_right_lane_at_center)), lane_color, 5)
        
        lane_boundaries.append((right_lane_x_center, int(h - right_lane_at_center)))
        lane_boundaries.append((right_lane_x_intercept, h))
        
    # color in between
    lane = np.zeros_like(frame)
    lane_vertices = np.array([lane_boundaries], dtype=np.int32)
    cv2.fillPoly(lane, lane_vertices, lane_fill_color)
    
    # merge image
    output_frame = cv2.addWeighted(lane, lane_fill_opacity, frame, 1, 0)
    
    
    # build and display video
    cv2.imshow("Lane Detection", output_frame)
    output.write(output_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
output.release()
cap.release()
