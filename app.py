import time
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIGURATION & STATE
# ---------------------------------------------------------------------------
# Calibrated table zones for 680x342 resolution video window
TABLE_ZONES = {
    "Table 1": np.array([[120, 240], [240, 240], [240, 330], [120, 330]], np.int32),
    "Table 2": np.array([[260, 230], [390, 230], [390, 320], [260, 320]], np.int32),
    "Table 3": np.array([[410, 210], [560, 210], [560, 310], [410, 310]], np.int32),
}

# Table Capacities
TABLE_CAPACITIES = {
    "Table 1": 4,
    "Table 2": 2,
    "Table 3": 4,
}

# State Tracking per table
table_states = {
    name: {
        "occupied": False,
        "start_time": None,
        "guest_count": 0,
        "raw_counts": [],  # For temporal smoothing
    }
    for name in TABLE_ZONES
}

SMOOTHING_FRAMES = 15  # Number of frames to average posture/presence over


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def is_seated_pose(keypoints):
    """Determines if a person is seated using pose keypoints."""
    if keypoints is None or len(keypoints) == 0:
        return False

    kp = keypoints[0]  # First person keypoint tensor

    try:
        l_hip, r_hip = kp[11], kp[12]
        l_knee, r_knee = kp[13], kp[14]
        l_sh, r_sh = kp[5], kp[6]

        torso_h = abs((l_hip[1] + r_hip[1]) / 2 - (l_sh[1] + r_sh[1]) / 2)
        thigh_h = abs((l_knee[1] + r_knee[1]) / 2 - (l_hip[1] + r_hip[1]) / 2)

        if torso_h > 0:
            ratio = thigh_h / torso_h
            if ratio < 0.6:  # Smaller vertical thigh distance = sitting/bent legs
                return True
    except Exception:
        pass

    return False


def format_timer(seconds):
    """Formats elapsed seconds into MM:SS format."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    print("Loading YOLOv8 Pose model...")
    model = YOLO("yolov8n-pose.pt")

    video_filename = "video_1.mp4"
    cap = cv2.VideoCapture(video_filename)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_filename}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Loaded Successfully! Resolution: {frame_width}x{frame_height}")
    print("Processing video. Press 'q' in the video window to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, verbose=False)[0]
        current_table_guests = {name: 0 for name in TABLE_ZONES}

        if results.keypoints is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            keypoints_data = results.keypoints.data.cpu().numpy()

            for box, kps in zip(boxes, keypoints_data):
                px = int((box[0] + box[2]) / 2)
                py = int((box[1] + box[3]) / 2)

                seated = is_seated_pose(kps)

                # Fallback heuristic: Box aspect ratio when keypoints are hidden
                w = box[2] - box[0]
                h = box[3] - box[1]
                if not seated and (w / h) > 0.75:
                    seated = True

                if seated:
                    for zone_name, polygon in TABLE_ZONES.items():
                        if cv2.pointPolygonTest(polygon, (px, py), False) >= 0:
                            current_table_guests[zone_name] += 1

        # Update table states with smoothing
        total_seated = 0
        total_capacity = sum(TABLE_CAPACITIES.values())

        for name, data in table_states.items():
            data["raw_counts"].append(current_table_guests[name])
            if len(data["raw_counts"]) > SMOOTHING_FRAMES:
                data["raw_counts"].pop(0)

            avg_count = int(round(np.mean(data["raw_counts"])))
            data["guest_count"] = avg_count

            if avg_count > 0:
                if not data["occupied"]:
                    data["occupied"] = True
                    data["start_time"] = time.time()
                total_seated += avg_count
            else:
                data["occupied"] = False
                data["start_time"] = None

        # -------------------------------------------------------------------
        # RENDERING OVERLAYS
        # -------------------------------------------------------------------
        overlay = frame.copy()

        for name, polygon in TABLE_ZONES.items():
            data = table_states[name]
            is_occ = data["occupied"]

            color = (0, 0, 255) if is_occ else (0, 255, 0)
            cv2.fillPoly(overlay, [polygon], color)

            x, y, w, h = cv2.boundingRect(polygon)
            card_text = f"{name}: FREE"
            if is_occ:
                elapsed = time.time() - data["start_time"]
                card_text = f"{data['guest_count']} Guests {format_timer(elapsed)}"

            cv2.putText(
                frame,
                card_text,
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        for name, polygon in TABLE_ZONES.items():
            color = (0, 0, 255) if table_states[name]["occupied"] else (0, 255, 0)
            cv2.polylines(frame, [polygon], True, color, 2)

        # Top Dashboard Summary Bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (30, 30, 30), -1)
        dash_text = f"Capacity: {total_seated}/{total_capacity} Seated | Active Tables: {sum(1 for d in table_states.values() if d['occupied'])}/{len(TABLE_ZONES)}"
        cv2.putText(
            frame,
            dash_text,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Restaurant Occupancy & Timer Tracker", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()