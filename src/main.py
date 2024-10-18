import os
import cv2
from camera import Camera
from face_detection import FaceDetector
from distance_estimation import calculate_distance

# Set the path for the Haar Cascade files
cascade_path_face = 'data/haarcascade_frontalface_default.xml'
cascade_path_eye = 'data/haarcascade_eye.xml'

if not os.path.exists(cascade_path_face) or not os.path.exists(cascade_path_eye):
    print("Error: Cascade files not found")
    exit(1)

# Create instances of FaceDetector and Eye Detector
face_detector = FaceDetector(cascade_path_face)
eye_cascade = cv2.CascadeClassifier(cascade_path_eye)

# Define the actual width of the face in centimeters
actual_width = 17.0  

# Define the camera's focal length in pixels
focal_length = 500.0  

# Initialize the camera for capturing frames
camera = Camera()

# Define minimum and maximum face size
min_face_size = (300, 300)  # Minimum width and height
max_face_size = (1000, 1000)  # Maximum width and height

try:
    while True:
        # Capture a frame from the camera
        frame = camera.read_frame()

        # Detect faces in the captured frame
        faces = face_detector.detect_faces(frame)

        # Initialize eyes variable to check later
        eyes_detected = False

        # Process and display all valid faces
        for (x, y, w, h) in faces:
            if (min_face_size[0] <= w <= max_face_size[0]) and (min_face_size[1] <= h <= max_face_size[1]):
                distance = calculate_distance(w, focal_length, actual_width)
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Region of Interest for eyes detection
                roi_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

                # Detect eyes within the face region
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) > 0:  # Only proceed if eyes are detected
                    eyes_detected = True  # Mark that eyes are detected

                    # Sort eyes based on area (width * height) and select the two largest
                    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

                    # Draw a circle around each of the two largest detected eyes
                    for (ex, ey, ew, eh) in eyes:
                        eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                        radius = int(0.3 * (ew + eh) // 2)  # Adjust radius as needed
                        cv2.circle(frame, eye_center, radius, (0, 255, 0), 2)

                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Show the distance only when eyes are detected
                    cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show the processed frame only if at least one eye is detected
        if eyes_detected:
            cv2.imshow('Eye Detection', frame)

        # Exit the loop when 'q' or 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF in {ord('q'), 27}:
            break

finally:
    # Clean up: release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
    print("Program terminated gracefully.")
