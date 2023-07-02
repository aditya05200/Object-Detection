#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

# Load the pre-trained YOLOv4-tiny model
net = cv2.dnn.readNet(r"C:\Users\adity\Downloads\yolov4-tiny.weights", r"C:\Users\adity\Downloads\yolov4-tiny.cfg")

# Load the class labels
classes = []
with open(r"C:\Users\adity\Downloads\classes.txt", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Set up the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create a window
cv2.namedWindow("Object Detection")

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # Process the outputs
    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Apply non-maximum suppression to remove overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.1)

    # Draw the bounding boxes and labels
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Check for key press
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows(q)

