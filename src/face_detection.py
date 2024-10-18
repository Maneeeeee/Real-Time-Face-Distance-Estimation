import cv2

class FaceDetector:
    def __init__(self, cascade_path):
        '''Initialize the face cascade classifier with the given path'''
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, frame):
        '''Convert the captured frame to grayscale for face detection'''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use the cascade to detect faces in the grayscale image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces




