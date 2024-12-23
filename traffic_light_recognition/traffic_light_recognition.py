import cv2
from ultralytics import YOLO
from enum import Enum
import numpy as np

# Load the models
coco = YOLO('yolov8s.pt')
oi7 = YOLO('yolov8s-oiv7.pt')

# Open the image
image = cv2.imread("traffic_lights.jpg")

# Colors
red = (0, 0, 255)
yellow = (0, 255, 255)
green = (0, 255, 0)
colors = [red, yellow, green]

# Run inference on the image to get boxes
coco_results = coco.predict(image, conf=0.33, classes=[0,1,2,3,5,7,9,11,12])
# person, bicycle, motorcycle, car, bus, truck, traffic light, stop sign, parking meter
annotated_image = coco_results[0].plot()

oi7_results = oi7.predict(annotated_image, conf=0.33, classes=[6,42,73,90,342,370,381,495,522,548,549,558])
# ambulance, bicycle, bus, car, motorcycle, parking meter, person, stop sign, taxi, traffic light, traffic sign, truck
output_image = oi7_results[0].plot()

boxes = coco_results[0].boxes.xyxy.tolist() + oi7_results[0].boxes.xyxy.tolist()
classifications = coco_results[0].boxes.cls.tolist() + oi7_results[0].boxes.cls.tolist()
    
def argmax(arr):
    return max(range(len(arr)), key=lambda x : arr[x])

# Determine the color of the traffic lights
for i in range(len(boxes)):
    box = boxes[i]
    classification = classifications[i]
    
    if classification != 9 and classification != 548: # skip if not traffic light
        continue
    
    # Select regions of interest
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x_padding = (x2 - x1) // 8
    y_padding = (y2 - y1) // 8
    offset = (y2 - y1) / 3
    traffic_lights = [image[y1+y_padding:int(y1-y_padding+offset),x1+x_padding:x2-x_padding], image[int(y1+y_padding+offset):int(y1-y_padding+2*offset),x1+x_padding:x2-x_padding], image[int(y1+y_padding+2*offset):y2-y_padding,x1+x_padding:x2-x_padding]]
    
    intensities = [np.average(traffic_light) for traffic_light in traffic_lights]
    closest_color = colors[argmax(intensities)]

    output_image = cv2.rectangle(output_image, (x1,y1), (x2,y2), closest_color, 10)

# Annotate and save the image
cv2.imwrite("traffic_lights_inference.png", output_image)
