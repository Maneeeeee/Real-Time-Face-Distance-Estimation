import cv2

class FaceDetector:
    def __init__(self, cascade_path, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        '''Initialize the face cascade classifier with the given path'''
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load cascade classifier")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect_faces(self, frame):
        '''Convert the captured frame to grayscale for face detection'''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use the cascade to detect faces in the grayscale image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=self.min_size)
        return faces
