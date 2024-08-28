import cv2
from ultralytics import YOLO
from enum import Enum
import numpy as np

# Load the models
coco = YOLO('yolov8s.pt')
oi7 = YOLO('yolov8s-oiv7.pt')

# Open the image
image = cv2.imread("red_traffic.jpeg.webp")

# Colors
red = (0, 0, 255)
yellow = (0, 255, 255)
green = (0, 255, 0)
colors = [red, yellow, green]
gray = (128, 128, 128)

# Run inference on the image to get boxes
coco_results = coco.predict(image, conf=0.33, classes=[0,1,2,3,5,7,9,11,12])
# person, bicycle, motorcycle, car, bus, truck, traffic light, stop sign, parking meter
annotated_image = coco_results[0].plot()

oi7_results = oi7.predict(annotated_image, conf=0.33, classes=[6,42,73,90,342,370,381,495,522,548,549,558])
# ambulance, bicycle, bus, car, motorcycle, parking meter, person, stop sign, taxi, traffic light, traffic sign, truck
output_image = oi7_results[0].plot()

boxes = coco_results[0].boxes.xyxy.tolist() + oi7_results[0].boxes.xyxy.tolist()
classifications = coco_results[0].boxes.cls.tolist() + oi7_results[0].boxes.cls.tolist()

def normalize(color):
    return (color - np.min(color)) / (np.max(color) - np.min(color)) * 255

def distance(color1, color2):
    return np.linalg.norm(normalize(color1) - normalize(color2))
    
def argmin(arr):
    return min(range(len(arr)), key=lambda x : arr[x])

# Determine the color of the traffic lights
for i in range(len(boxes)):
    box = boxes[i]
    classification = classifications[i]
    
    if classification != 9 and classification != 548: # skip if not traffic light
        continue
    
    # Select region of interest
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    traffic_light = image[y1:y2,x1:x2]
    
    # Flatten
    width, height, _ = traffic_light.shape
    flattened_traffic_light = traffic_light.reshape((width * height, 3))
    
    # Filter dimmest pixels
    average_b, average_g, average_r = np.mean(flattened_traffic_light, axis=0)
    filtered_traffic_light = flattened_traffic_light[(flattened_traffic_light[:,0] > average_b) | (flattened_traffic_light[:,1] > average_g) | (flattened_traffic_light[:,2] > average_r)]
    average_color = np.mean(filtered_traffic_light, axis=0)
    
    # Classify color
    color_distances = [distance(average_color, colors[0]), distance(average_color, colors[1]), distance(average_color, colors[2])]
    closest_color = colors[argmin(color_distances)]

    output_image = cv2.rectangle(output_image, (x1,y1), (x2,y2), closest_color, 10)

# Annotate and save the image
cv2.imwrite("red_traffic_inference.png", output_image)
