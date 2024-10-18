import os
import cv2
from camera import Camera
from face_detection import FaceDetector
from distance_estimation import calculate_distance

# Set the path for the Haar Cascade file
cascade_path = 'data/haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"Error: {cascade_path} not found")
    exit(1)

# Create an instance of FaceDetector
face_detector = FaceDetector(cascade_path)

# Define the actual width of the face in centimeters (average value i've choose 17cm)
actual_width = 17.0  


# Define the camera's focal length in pixels (for my camera it's 500 pixels)

focal_length = 500.0  # Adjust based on your camera's specifications

# Initialize the camera for capturing frames
camera = Camera()

while True:
    # Capture a frame from the camera
    frame = camera.read_frame()

    # Detect faces in the captured frame and calculate distances
    faces = face_detector.detect_faces(frame)
    for (x, y, w, h) in faces:
        distance = calculate_distance(w, focal_length, actual_width)
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the calculated distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show the processed frame with distance information
    cv2.imshow('Distance Measurement', frame)

    # Exit the loop when 'q' or 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF in {ord('q'), 27}:
        break

# Clean up: release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
