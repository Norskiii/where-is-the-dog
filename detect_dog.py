import cv2
import numpy as np
import os

##### DEFINE WEIGHT AND MODEL FILES HERE #####
weights = "bin/yolov3.weights"
model = "cfg/yolov3.cfg"
##############################################

cwd = os.getcwd()
weights_dir = os.path.join(cwd, weights)
model_dir = os.path.join(cwd, model)

net = cv2.dnn.readNet(weights_dir, model_dir)

classes = []

with open("coco.names", "r") as f:    
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

##### INPUT IMAGE #####
image = cv2.imread("dog1.jpg")

font = cv2.FONT_HERSHEY_PLAIN

height, width, channels = image.shape

# Detect objects
blob = cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

class_id = 0
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = float(scores[class_id])
        
        if class_id == 16:
            # Possible dog detected!
            if confidence > 0.5:
                w = detection[2] * width
                h = detection[3] * height
                x = detection[0] * width - np.rint(w/2)
                y = detection[1] * height + np.rint(h/2)

                boxes.append([int(x), int(y), int(w), int(h)])
                confidences.append(confidence)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

if len(boxes) == 0:
    print("No dogs found :(")

else:
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "dog"
            confidence = confidences[i]
            color = (0, 255, 0)

            cv2.rectangle(image, (x,y), (x+w, y-h), (0,255,0), 2)
            cv2.putText(image, label + " " + str(round(confidence,2)), (x, y + 30), font, 3, color, 3)
   
    cv2.imshow("Image", image)
    cv2.waitKey(0)
