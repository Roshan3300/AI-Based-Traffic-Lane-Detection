import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# LOAD VIDEO
# -----------------------------
cap = cv2.VideoCapture("lan.mp4")

if not cap.isOpened():
    print("❌ Video not found")
    exit()

# -----------------------------
# LOAD YOLO MODEL (DL)
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,480))
    display = frame.copy()

    # ======================================================
    # 1️⃣ OBJECT DETECTION (DEEP LEARNING)
    # ======================================================
    vehicles = []

    results = model(frame, imgsz=320, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if conf > 0.4 and label in ["car","truck","bus","motorcycle"]:
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                cx = (x1+x2)//2
                vehicles.append(cx)

                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,255),2)
                cv2.putText(display,label,(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

    # ======================================================
    # 2️⃣ LANE DETECTION (CV)
    # ======================================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)

    histogram = np.sum(edges[edges.shape[0]//2:, :], axis=0)

    midpoint = histogram.shape[0]//2
    left = np.argmax(histogram[:midpoint])
    right = np.argmax(histogram[midpoint:]) + midpoint

    lane_center = (left + right) // 2
    car_center = 320

    offset = car_center - lane_center

    # ======================================================
    # 3️⃣ NLP-STYLE DECISION ENGINE (SIMULATED)
    # ======================================================
    decision = "Keep Lane"
    reasoning = "Road is clear"

    if len(vehicles) > 0:
        avg_vehicle = sum(vehicles)/len(vehicles)

        if avg_vehicle < 320:
            decision = "Move Right"
            reasoning = "Vehicle detected on left side"
        else:
            decision = "Move Left"
            reasoning = "Vehicle detected on right side"

    elif abs(offset) > 30:
        if offset > 0:
            decision = "Steer Left"
            reasoning = "Car drifting right from lane center"
        else:
            decision = "Steer Right"
            reasoning = "Car drifting left from lane center"

    # ======================================================
    # 4️⃣ VISUALIZATION
    # ======================================================

    # lane center line
    cv2.line(display, (lane_center,480),(lane_center,300),(0,255,0),2)

    # car center
    cv2.line(display, (320,480),(320,300),(255,0,0),2)

    # steering arrow
    angle = np.clip(offset, -100, 100) / 2
    end_x = int(320 + angle)
    end_y = 380

    cv2.arrowedLine(display,(320,480),(end_x,end_y),(0,0,255),3)

    # ======================================================
    # 5️⃣ SHOW "AI THINKING" ON SCREEN
    # ======================================================
    cv2.rectangle(display, (10,10), (630,140), (0,0,0), -1)

    cv2.putText(display, "AI DECISION SYSTEM",
                (20,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(display, f"Decision: {decision}",
                (20,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    cv2.putText(display, f"Reason: {reasoning}",
                (20,90), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    cv2.putText(display, f"Lane Offset: {offset}",
                (20,120), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    # ======================================================
    # SHOW OUTPUT
    # ======================================================
    cv2.imshow("Self Driving AI Demo", display)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()