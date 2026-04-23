from ultralytics import YOLO
import cv2
import serial
import time

model = YOLO("best_1.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

arduino = serial.Serial('COM3', 9600, timeout=0.1)
time.sleep(2)

# ── Config ─────────────────────────────────────────────────────
COOLDOWN_SECONDS  = 10
CONFIRM_FRAMES    = 5       # Frames in a row before accepting detection
CONFIDENCE_THRESH = 0.75    # Minimum confidence to consider a detection

last_sent_time    = 0
last_sent_command = None
detection_counter = {}

def is_cooling_down():
    return (time.time() - last_sent_time) < COOLDOWN_SECONDS

def seconds_remaining():
    return max(0, COOLDOWN_SECONDS - (time.time() - last_sent_time))

def drain_serial():
    while arduino.in_waiting > 0:
        line = arduino.readline().strip()
        if line:
            print(f"  [Arduino] {line.decode(errors='ignore')}")

def send_command(command: bytes):
    global last_sent_time, last_sent_command
    arduino.write(command)
    last_sent_time = time.time()
    last_sent_command = command
    print(f"  [Sent] {command.decode().strip()} — cooling down for {COOLDOWN_SECONDS}s")

print("System ready. Scanning for waste...")

# ── Main Loop ──────────────────────────────────────────────────
while True:
    drain_serial()

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    results = model(frame)

    detected_this_frame = set()

    for result in results:
        class_ids   = result.boxes.cls
        confidences = result.boxes.conf
        class_names = [model.names[int(cls_id)] for cls_id in class_ids]

        # ── Filter 1: Confidence threshold ─────────────────────
        for name, conf in zip(class_names, confidences):
            if conf < CONFIDENCE_THRESH:
                print(f"  [Ignored] {name} — confidence too low ({conf:.0%})")
                continue
            if name in ("Plastic", "Paper"):
                detected_this_frame.add(name)

        # Draw boxes only for confident detections
        for box, name, conf in zip(result.boxes.xyxy, class_names, confidences):
            if conf < CONFIDENCE_THRESH:
                continue
            x1, y1, x2, y2 = box
            label = f"{name} {conf:.0%}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ── Filter 2: Consistent frame counter ─────────────────────
    for name in detected_this_frame:
        detection_counter[name] = detection_counter.get(name, 0) + 1

    # Reset counter for classes not seen this frame
    for name in list(detection_counter):
        if name not in detected_this_frame:
            detection_counter[name] = 0

    # ── Filter 3: Cooldown check before sending ─────────────────
    for name, count in detection_counter.items():
        if count >= CONFIRM_FRAMES and not is_cooling_down():
            command = b'plastic\n' if name == "Plastic" else b'paper\n'
            send_command(command)
            detection_counter.clear()
            break

    # ── Overlay UI ─────────────────────────────────────────────
    if is_cooling_down():
        secs = int(seconds_remaining()) + 1
        status_text  = f"Sorting... next scan in {secs}s"
        status_color = (0, 60, 220)
    else:
        status_text  = "Ready to scan"
        status_color = (0, 200, 0)

    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if last_sent_command:
        last_label = last_sent_command.decode().strip().upper()
        cv2.putText(frame, f"Last sent: {last_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

    # Show confirmation progress for each detected class
    for i, (name, count) in enumerate(detection_counter.items()):
        if count > 0:
            progress = f"Confirming {name}: {count}/{CONFIRM_FRAMES}"
            cv2.putText(frame, progress, (10, 90 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

    cv2.imshow('Eco Waste Sorter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()