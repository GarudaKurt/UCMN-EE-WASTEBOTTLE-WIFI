from ultralytics import YOLO #pip install ultralytics
import cv2
import serial
import time
#colab google train ML
model = YOLO("best_1.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open webcam.")
    exit()

# arduino = serial.Serial('COM5', 9600)
time.sleep(2)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    results = model(frame)

    for result in results:
        class_ids = result.boxes.cls
        class_names = [model.names[int(cls_id)] for cls_id in class_ids]

        for name in class_names:
            if name == "Plastic":
                # arduino.write(b'plastic\n')
                print("Plastic detected. Command sent to Arduino: plastic")
            elif name == "Paper":
                # arduino.write(b'paper\n')
                print("Paper detected. Command sent to Arduino: paper")

        for box, name in zip(result.boxes.xyxy, class_names):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# arduino.close()
cv2.destroyAllWindows()