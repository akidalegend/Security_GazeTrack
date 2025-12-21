# Smart Security Camera System

This project implements a basic smart security camera system using Python and OpenCV. It uses background subtraction to detect motion in a video feed.

## Prerequisites

- Python 3.x
- A webcam or video file

## Installation

1.  Clone the repository or download the files.
2.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script using Python:

```bash
python security_cam.py
```

The script will open a window showing the camera feed. When motion is detected, a green bounding box will be drawn around the moving object, and "Motion Detected" text will appear.

A second window will show the "Motion Mask", which visualizes the background subtraction process (white pixels represent motion).

Press `q` to quit the application.

## Configuration

You can adjust the sensitivity of the motion detection by modifying the `security_cam.py` file:

-   `history`: Increase to make the background model update slower.
-   `varThreshold`: Increase to make the detection less sensitive to noise.
-   `cv2.contourArea(contour) < 500`: Adjust the `500` value to filter out smaller or larger objects.
