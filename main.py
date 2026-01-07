from ultralytics import YOLO
import cv2
import numpy as np
import time
import requests

# 🔹 Your Pushbullet Access Token
PUSHBULLET_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # replace with your real token

def send_pushbullet_alert(message):
    """Send instant notification to your phone."""
    data = {"type": "note", "title": "⚠️ Accident Detected!", "body": message}
    resp = requests.post(
        "https://api.pushbullet.com/v2/pushes",
        json=data,
        headers={"Access-Token": PUSHBULLET_TOKEN}
    )
    if resp.status_code == 200:
        print("✅ Alert sent to phone!")
    else:
        print("❌ Failed to send alert:", resp.text)


# --- Load YOLO model ---
model = YOLO("yolov8m.pt")

# --- Load video ---
cap = cv2.VideoCapture("test_video.mp4")
if not cap.isOpened():
    print("❌ Error: Cannot open video file.")
    exit()

# --- IOU function ---
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


frame_count = 1
last_alert_time = 0
alert_cooldown = 3  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 2

    results = model(frame, verbose=False)
    boxes = []

    # Detect vehicles
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            label = model.names[cls]
            if label in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = box.xyxy[0]
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Check for collisions
    accident = False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            overlap = iou(boxes[i], boxes[j])
            if overlap > 0.6:  # Lower threshold = more sensitive
                accident = True
                cv2.putText(frame, " ACCIDENT DETECTED!", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                break
        if accident:
            break

    # Send Pushbullet alert
    if accident and (time.time() - last_alert_time > alert_cooldown):
        print("⚠️ Accident detected! Sending phone alert...")
        send_pushbullet_alert("An accident was detected in the live feed. Please check immediately!")
        last_alert_time = time.time()

    cv2.imshow("Road Accident Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
