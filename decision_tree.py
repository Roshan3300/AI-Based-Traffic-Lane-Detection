import cv2
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture("lan.mp4")

if not cap.isOpened():
    print("❌ Video not found")
    exit()

model = YOLO("yolov8n.pt")

# -----------------------------
# DRAW NEURAL NETWORK
# -----------------------------
def draw_nn(frame, inputs, hidden, output):

    base_x = 420
    base_y = 50

    # positions
    input_nodes = [(base_x, base_y+50), (base_x, base_y+150)]
    hidden_nodes = [(base_x+100, base_y+30),
                    (base_x+100, base_y+100),
                    (base_x+100, base_y+170)]
    output_node = (base_x+220, base_y+100)

    # draw connections
    for i, inp in enumerate(input_nodes):
        for j, hid in enumerate(hidden_nodes):
            weight = abs(inputs[i]) / 100
            color = (0, int(255*weight), 255-int(255*weight))
            cv2.line(frame, inp, hid, color, 2)

    for j, hid in enumerate(hidden_nodes):
        weight = abs(hidden[j])
        color = (0, int(255*weight), 255-int(255*weight))
        cv2.line(frame, hid, output_node, color, 2)

    # draw nodes
    for i, (x,y) in enumerate(input_nodes):
        val = min(abs(inputs[i])/100,1)
        cv2.circle(frame, (x,y), 15, (0,int(255*val),255), -1)
        cv2.putText(frame, f"I{i}", (x-10,y+5), 0, 0.4, (0,0,0),1)

    for j, (x,y) in enumerate(hidden_nodes):
        val = min(abs(hidden[j]),1)
        cv2.circle(frame, (x,y), 15, (0,int(255*val),255), -1)
        cv2.putText(frame, f"H{j}", (x-10,y+5), 0, 0.4, (0,0,0),1)

    val = min(abs(output),1)
    cv2.circle(frame, output_node, 20, (0,int(255*val),255), -1)
    cv2.putText(frame, "OUT", (output_node[0]-20,output_node[1]+5),
                0, 0.4, (0,0,0),1)

# -----------------------------
# SIMPLE NN FOR DECISION
# -----------------------------
def neural_network(offset, vehicle_pos):

    # normalize inputs
    x1 = offset / 100
    x2 = (vehicle_pos - 320) / 320 if vehicle_pos is not None else 0

    inputs = [x1, x2]

    # hidden layer (manual weights)
    h1 = np.tanh(0.6*x1 + 0.3*x2)
    h2 = np.tanh(-0.4*x1 + 0.8*x2)
    h3 = np.tanh(0.2*x1 - 0.5*x2)

    hidden = [h1, h2, h3]

    # output
    out = np.tanh(h1 + h2 - h3)

    return inputs, hidden, out

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,480))
    display = frame.copy()

    # ============================
    # VEHICLE DETECTION
    # ============================
    vehicle_center = None

    results = model(frame, imgsz=320, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if conf > 0.4 and label in ["car","truck","bus","motorcycle"]:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                vehicle_center = (x1+x2)//2
                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,255),2)

    # ============================
    # LANE CENTER
    # ============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)

    hist = np.sum(edges[edges.shape[0]//2:, :], axis=0)
    mid = hist.shape[0]//2

    left = np.argmax(hist[:mid])
    right = np.argmax(hist[mid:]) + mid

    lane_center = (left + right)//2
    offset = 320 - lane_center

    cv2.line(display,(lane_center,480),(lane_center,300),(0,255,0),2)

    # ============================
    # NN DECISION
    # ============================
    inputs, hidden, out = neural_network(offset, vehicle_center)

    if out > 0.3:
        decision = "Move Right"
    elif out < -0.3:
        decision = "Move Left"
    else:
        decision = "Keep Lane"

    # ============================
    # DRAW NN
    # ============================
    draw_nn(display, inputs, hidden, out)

    # ============================
    # TEXT
    # ============================
    cv2.putText(display, f"Decision: {decision}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(display, f"Offset: {offset}",
                (20,80), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    # ============================
    # SHOW
    # ============================
    cv2.imshow("Neural Self Driving AI", display)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()