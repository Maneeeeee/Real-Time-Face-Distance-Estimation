import cv2

class Camera:
    def __init__(self, camera_index=0):
        '''Initialize the camera using the specified index (default is 0 for the primary camera)'''
        self.cap = cv2.VideoCapture(camera_index)

    def read_frame(self):
        '''Capture a frame from the camera'''
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self):
        '''Release the camera resource when done'''
        self.cap.release()
