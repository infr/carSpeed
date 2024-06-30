import cv2
import numpy as np
import tensorflow as tf
import time
import datetime

# Load the TensorFlow model
model = tf.saved_model.load('/.cache/kagglehub/models/tensorflow/ssd-mobilenet-v2/tensorFlow2/fpnlite-320x320/1')

# Function to preprocess input frames
def preprocess(frame):
    frame_resized = cv2.resize(frame, (320, 320))
    return np.expand_dims(frame_resized, axis=0).astype('uint8')

# Initialize video capture
cap = cv2.VideoCapture(1)

# Define points A and B as fractions of the frame width
point_a = 0.39
point_b = 0.75

# Known distance between points A and B in meters
distance_meters = 20.0

timestamps_at_a = {}
timestamps_at_b = {}

# Initial box position and size
box_x = 660
box_y = 70
box_width = 890
box_height = 890


class_labels = {
    1: 'Person',
    2: 'Bicycle',
    3: 'Car',
    4: 'Motorcycle',
    6: 'Bus',
}

detected_objects = set()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_height, frame_width, _ = frame.shape


        # Crop the frame to the selected box area
    cropped_frame = frame[max(0, box_y):max(0, box_y) + box_height, max(0, box_x):max(0, box_x) + box_width]

    # Preprocess the cropped frame
    input_tensor = tf.convert_to_tensor(preprocess(cropped_frame))
    detections = model.signatures['serving_default'](input_tensor)

    point_a_x = int(frame_width * point_a)
    point_b_x = int(frame_width * point_b)

    cv2.line(frame, (point_a_x, 0), (point_a_x, frame_height), (255, 0, 0), 2)
    cv2.line(frame, (point_b_x, 0), (point_b_x, frame_height), (0, 255, 0), 2)
    
    detection_boxes = detections['detection_boxes'].numpy()[0]
    detection_scores = detections['detection_scores'].numpy()[0]
    detection_classes = detections['detection_classes'].numpy()[0]
    num_detections = int(detections['num_detections'].numpy()[0])

    # Draw the box on the frame
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 2)

    for i in range(num_detections):
        class_id = int(detection_classes[i])
        if detection_scores[i] > 0.5 and class_id in class_labels:
            detection_key = (class_id, i)  # Unique key for each detection
            if detection_key not in detected_objects:
                detected_objects.add(detection_key)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{timestamp} - {class_labels[class_id]} {i} detected.")
            
            box = detection_boxes[i]

            y_min, x_min, y_max, x_max = box  # Unpacking the normalized coordinates
            x_min, x_max = x_min * frame_width, x_max * frame_width
            y_min, y_max = y_min * frame_height, y_max * frame_height

            # Drawing bounding box around the detected object
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            
            x_center = (box[1] + box[3]) / 2 * frame_width
            cv2.putText(frame, f"{class_labels[class_id]} {i}", (int(x_center), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check if object crosses point A or B and calculate speed
            if x_center < frame_width * point_a:
                timestamps_at_a[i] = time.time()
            elif x_center > frame_width * point_b and i in timestamps_at_a:
                timestamps_at_b[i] = time.time()
                if i in timestamps_at_a:
                    time_passed = timestamps_at_b[i] - timestamps_at_a[i]
                    speed_mps = distance_meters / time_passed
                    speed_kmph = speed_mps * 3.6
                    print(f"{class_labels[class_id]} {i}: {speed_kmph:.2f} km/h")  # Use class_labels[class_id] instead of label
                    del timestamps_at_a[i]


    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('g'):
        point_a = max(0, point_a - 0.01)
    if key == ord('h'):
        point_a = min(1, point_a + 0.01)
    if key == ord('j'):
        point_b = max(0, point_b - 0.01)
    if key == ord('k'):
        point_b = min(1, point_b + 0.01)
    if key in [ord('g'), ord('h'), ord('j'), ord('k')]:
        print(f"Point A: {point_a:.2f}, Point B: {point_b:.2f}")

    # Move the box with WASD keys
    if key == ord('w'):
        box_y = max(box_y - 10, 0)
    elif key == ord('s'):
        box_y = min(box_y + 10, frame_height - box_height)
    elif key == ord('a'):
        box_x = max(box_x - 10, 0)
    elif key == ord('d'):
        box_x = min(box_x + 10, frame_width - box_width)

    # Resize the box with ZX keys
    if key == ord('z'):
        box_width = max(box_width - 10, 50)  # Minimum width
        box_height = max(box_height - 10, 50)  # Minimum height
    elif key == ord('x'):
        box_width = min(box_width + 10, frame_width - box_x)  # Maximum width
        box_height = min(box_height + 10, frame_height - box_y)  # Maximum height

    # Print the box position and size if WASD or ZX keys are pressed
    if key in [ord('w'), ord('s'), ord('a'), ord('d'), ord('z'), ord('x')]:
        print(f"Box Position and Size: X: {box_x}, Y: {box_y}, Width: {box_width}, Height: {box_height}")


cap.release()
cv2.destroyAllWindows()
