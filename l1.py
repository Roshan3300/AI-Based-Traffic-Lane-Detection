import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# VIDEO
# -----------------------------
vidcap = cv2.VideoCapture("lan.mp4")

if not vidcap.isOpened():
    print("❌ Cannot open video")
    exit()

model = YOLO("yolov8n.pt")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    success, frame = vidcap.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))
    display = frame.copy()

    # -----------------------------
    # ROI (ONLY ROAD AREA)
    # -----------------------------
    mask_roi = np.zeros_like(frame[:,:,0])

    polygon = np.array([[
        (0, 480),
        (640, 480),
        (500, 300),
        (140, 300)
    ]])

    cv2.fillPoly(mask_roi, polygon, 255)

    # -----------------------------
    # EDGE DETECTION
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    edges = cv2.bitwise_and(edges, mask_roi)

    # -----------------------------
    # GET LANE PIXELS
    # -----------------------------
    y, x = np.where(edges > 0)

    if len(x) > 1000:   # threshold removes noise

        midpoint = 320

        left_x = x[x < midpoint]
        left_y = y[x < midpoint]

        right_x = x[x >= midpoint]
        right_y = y[x >= midpoint]

        if len(left_x) > 200 and len(right_x) > 200:

            # -----------------------------
            # POLYNOMIAL FIT
            # -----------------------------
            left_fit = np.polyfit(left_y, left_x, 2)
            right_fit = np.polyfit(right_y, right_x, 2)

            plot_y = np.linspace(0, 479, 480)

            left_curve = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
            right_curve = right_fit[0]*plot_y**2 + right_fit[1]*plot_y + right_fit[2]

            # -----------------------------
            # DRAW LANE AREA
            # -----------------------------
            pts_left = np.array([np.transpose(np.vstack([left_curve, plot_y]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_curve, plot_y])))])

            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(display, np.int32([pts]), (0,255,100))

            # -----------------------------
            # STEERING
            # -----------------------------
            lane_center = (left_curve[-1] + right_curve[-1]) / 2
            offset = 320 - lane_center

            steering_angle = np.arctan(offset / 200) * 180 / np.pi

            end_x = int(320 + 100*np.sin(np.radians(steering_angle)))
            end_y = int(480 - 100*np.cos(np.radians(steering_angle)))

            cv2.line(display, (320,480), (end_x,end_y), (255,0,0), 3)

            cv2.putText(display, f"Steering: {steering_angle:.2f}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,255,255), 2)

    # -----------------------------
    # YOLO (optional, safe)
    # -----------------------------
    try:
        results = model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                if conf > 0.4 and label in ["car","truck","bus","motorcycle"]:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display, (x1,y1),(x2,y2),(0,255,255),2)
    except:
        pass

    # -----------------------------
    # SHOW
    # -----------------------------
    cv2.imshow("Output", display)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) == 27:
        break

vidcap.release()
cv2.destroyAllWindows()