import os
import cv2
from camera import Camera
from face_detection import FaceDetector
from distance_estimation import calculate_distance

# Set the path for the Haar Cascade files
cascade_path_face = 'data/haarcascade_frontalface_default.xml'

if not os.path.exists(cascade_path_face):
    print("Error: Cascade file not found")
    exit(1)

# Create an instance of FaceDetector
face_detector = FaceDetector(cascade_path_face)

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

        # Process and display all valid faces
        for (x, y, w, h) in faces:
            if (min_face_size[0] <= w <= max_face_size[0]) and (min_face_size[1] <= h <= max_face_size[1]):
                distance = calculate_distance(w, focal_length, actual_width)
                
                # Dynamic text size and color based on distance
                font_scale = 1.0 if distance > 50 else 1.5  # Increase size for closer distances
                color = (0, 255, 0) if distance > 50 else (0, 0, 255)  # Green for safe, Red for close

                cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show the processed frame
        cv2.imshow('Face Detection', frame)

        # Exit the loop when 'q' or 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF in {ord('q'), 27}:
            break

finally:
    # Clean up: release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
    print("Program terminated gracefully.")
