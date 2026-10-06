import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Store motion information
motion_data = []

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

print("================================")
print("     MOTION DETECTION SYSTEM")
print("================================")
print("Camera Status : ON")
print("Press Q to stop")

# Read first frame
ret, previous_frame = cap.read()

if not ret:
    print("Error: Could not read camera.")
    cap.release()
    exit()

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Compare frames
    difference = cv2.absdiff(previous_gray, gray)

    # Threshold
    threshold = cv2.threshold(
        difference, 25, 255, cv2.THRESH_BINARY
    )[1]

    threshold = cv2.dilate(threshold, None, iterations=2)

    # Find moving objects
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:

        if cv2.contourArea(contour) < 1000:
            continue

        motion_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Date and time
    now = datetime.now()

    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")

    if motion_detected:
        status = "MOTION DETECTED"
        color = (0, 0, 255)
    else:
        status = "NO MOTION"
        color = (0, 255, 0)

    # Display information
    cv2.putText(
        frame,
        "Camera Status : ON",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Date : " + date,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Time : " + time,
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Motion Status : " + status,
        (10, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Store motion information
    motion_data.append({
        "date": date,
        "time": time,
        "motion": 1 if motion_detected else 0
    })

    # Show camera
    cv2.imshow("Motion Detection System", frame)

    # Update previous frame
    previous_gray = gray

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Close camera
cap.release()
cv2.destroyAllWindows()


# Save JSON
with open("motion_data.json", "w") as file:
    json.dump(motion_data, file, indent=4)

print("\nMotion data saved to motion_data.json")


# Pandas analysis
df = pd.DataFrame(motion_data)

if not df.empty:

    total_frames = len(df)
    motion_frames = df["motion"].sum()
    no_motion_frames = total_frames - motion_frames

    print("\n================================")
    print("       MOTION STATISTICS")
    print("================================")
    print("Total Frames      :", total_frames)
    print("Motion Frames     :", motion_frames)
    print("No Motion Frames  :", no_motion_frames)

    # Generate graph
    plt.figure(figsize=(7, 5))

    plt.bar(
        ["Motion", "No Motion"],
        [motion_frames, no_motion_frames],
        color=["red", "green"]
    )

    plt.title("Motion Detection Statistics")
    plt.xlabel("Status")
    plt.ylabel("Number of Frames")

    plt.savefig("motion_statistics.png")

    plt.show()

else:
    print("No motion data available.")

print("Program completed.")