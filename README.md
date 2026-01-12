# Smart Security Camera System with Eye Tracking

This project implements an advanced smart security camera system using Python, OpenCV, and YOLO object detection, now featuring real-time eye tracking capabilities.

## Features

- **Motion Detection**: Background subtraction with configurable ROI (Region of Interest)
- **Object Detection**: YOLOv8 for person and object recognition
- **Eye Tracking**: Real-time gaze tracking with pupil detection and direction analysis
- **Recording**: Automatic recording when motion is detected
- **Dark Mode GUI**: Professional interface with system logs and recording management

## Prerequisites

- Python 3.x
- A webcam
- macOS, Windows, or Linux

## Installation

1.  Clone the repository or download the files.
2.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```
    
3.  The eye tracking model (`shape_predictor_68_face_landmarks.dat`) is already included in the `gaze_tracking/trained_models/` folder.

## Usage

Run the security camera system:

```bash
python security_cam.py
```

### Features:

- **Enable Restricted Zone (ROI)**: Focus motion detection on a specific area
- **Enable Eye Tracking**: Toggle real-time gaze tracking overlay
- **Motion Threshold**: Adjust sensitivity slider (500-5000)
- **System Log**: View real-time events
- **Recordings**: Play back recorded incidents

Press the "SHUT DOWN SYSTEM" button to exit.

## Eye Tracking Capabilities

When enabled, the eye tracking feature provides:
- **Pupil Detection**: Green circles highlight detected pupils
- **Gaze Direction**: Shows if looking LEFT, RIGHT, CENTER
- **Blink Detection**: Identifies when eyes are closed
- **Real-time Overlay**: All data displayed on the video feed

## Configuration

Adjust settings in `security_cam.py`:

-   `history=500`: Background model update speed
-   `varThreshold=50`: Motion detection sensitivity
-   `min_area`: Minimum motion contour size (controlled by slider)
-   `recording_cooldown=30`: Frames without motion before stopping recording
