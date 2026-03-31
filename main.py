from ultralytics import YOLO    
import cv2
import serial
import time


def main():
    # Setup the YOLO model and camera
    model = YOLO("best.pt")
    video = cv2.VideoCapture(0)

    # Setup the video output
    output_file = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Setup serial communication
    # arduino = serial.Serial('COM5', 9600)  # Replace COM_PORT according to your available CH340 port in device manager

    last_sent_time = time.time()  # Track the last time serial data was sent
    send_interval = 5  # Interval in seconds to send data

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=640, conf=0.4)

        class_name_mapping = {"waterbottle": "bottle", "5": "can"}
        annotated_frame = frame.copy()
        detected_class = None  # Variable to store the detected class to send to arduino

        # Process detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                if class_name in class_name_mapping:
                    class_name = class_name_mapping[class_name]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Store the detected class
                detected_class = class_name

        video_writer.write(annotated_frame)
        cv2.imshow("YOLO Predictions", annotated_frame)

        current_time = time.time()
        elapsed_time = current_time - last_sent_time

        if elapsed_time >= send_interval:  # Check if it's time to send data
            if detected_class:  # If a class is detected, send it to Arduino
                # arduino.write(detected_class.encode())  # Send detected class to Arduino
                print(f"Sent: {detected_class}")  # Optional: Print what is sent
            last_sent_time = current_time  # Reset the last sent time

        # Exit condition x1when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_file}")

if __name__ == "__main__":
    main()