from ultralytics import YOLO
import cv2


# Load the YOLO model
model = YOLO("yolov8m.pt")

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam, you can try different numbers for external webcams

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Couldn't open webcam.")
else:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Perform object detection on the frame
        results = model.predict(frame, show=True)
        print(results)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()