# Real-Time Face Detection 

This project uses OpenCV to detect **faces** in real-time using a webcam. It also calculates the distance of the detected face from the camera based on the size of the face in the captured frame.

## Features

- **Real-time face detection** using Haar Cascade Classifiers
- Distance estimation based on face size in the frame
- Dynamic display of distance with varying colors and text size based on proximity
- **Graceful termination** of the program

## Prerequisites

Ensure you have the following installed:

- **Python 3.x**
- **OpenCV (cv2 library)**

### Installation

1. **Clone this repository**:
    ```bash
    git clone https://github.com/Maneeeeee/Real-Time-Face-Distance-Estimation
    cd face_detection
    ```

2. **Install the required Python packages**:
    ```bash
    pip install -r data/requirements.txt
    ```
## Additional Information
I added a no_eye_detection.py file, which is faster but does not include eye detection and may falsely detect some faces in the background. I recommend users run the corresponding file based on their computer's performance:

1.If your computer is slow, use no_eye_detection.py.

2.If your computer is faster, use the original detection file (main.py).


Feel free to experiment with both options to see which one works best for your setup!

## How to Run

To run the project, use the following command:
```bash
python src/main.py
