from pathlib import Path

import cv2
import numpy as np


def get_video_reader(path):
    if isinstance(path, str):
        if Path(path).suffix in [".mp4"]:  # , ".avi", ".mov"]:
            return read_video
        else:
            return None
    else:
        return None


def read_video(path):
    # Open the video file
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        # Stop if we have reached the end of the video
        if not ret:
            break
        # Convert frame to RGB (if necessary)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    # Release the video capture object
    cap.release()
    data = np.stack(frames, axis=0, dtype=np.uint8)
    add_kwargs = {
        "name": Path(path).stem,
        "metadata": {"path": path},
    }
    return [(data, add_kwargs, "image")]
