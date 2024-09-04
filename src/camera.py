import cv2  # Lib containing imaging functions and other stuff
import serial
import time

door_open = False
arduino_connected = False

# Attempts to connect to an arduino which is configured to move a weighted servo motor to open door
try:
    arduino = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino's port
    arduino_connected = True
    print("Arduino connected successfully.")
except (serial.SerialException, FileNotFoundError):  # Created because my arduino gonked up
    print("Arduino not connected. Running in mock mode.")


# Initialize camera or print if stuff messed up
def initialize_camera():
    camera = cv2.VideoCapture(0)  # opens systems default camera hence 0
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return None
    else:
        print("Camera initialized successfully.")
        return camera


# Loads the face recognizer model
def load_face_recognizer(model_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)
    return face_recognizer


# Opens the door if the confidence > required confidence threshold by pressing O
def trigger_door_open():
    global door_open
    if not door_open:
        if arduino_connected:
            print("Sending open command to Arduino...")
            arduino.write(b'O')  # Send the 'O' command to open the door
        else:
            print("[MOCK] Arduino would open the door now...")
        door_open = True
    else:
        print("Door is already open.")


# Closes the door if the confidence > required confidence threshold by pressing C
def trigger_door_close():
    global door_open
    if door_open:
        if arduino_connected:
            print("Sending close command to Arduino...")
            arduino.write(b'C')  # Send the 'C' command to close the door
        else:
            print("[MOCK] Arduino would close the door now...")
        door_open = False
    else:
        print("Door is already closed.")


# Initializes video feed if camera and frame are working
def display_video_feed(camera, face_recognizer, label_map, target_label_id, open_door_threshold=50):
    if camera is None:
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        recognized_person = False  # Reset flag for every frame

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_roi = gray[y:y + h, x:x + w]
            label_id, confidence = face_recognizer.predict(face_roi)

            if confidence > open_door_threshold and label_id == target_label_id:
                label_text = f"{label_map[label_id]} ({int(confidence)})"
                recognized_person = True  # Set the flag to True if face is recognized
            else:
                label_text = "Unknown"

            # Display the recognized label or "Unknown"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # print(label_text + "PROBLEM HERE")
        cv2.imshow("Video Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the program with 'q'
            # print("QUIT NOT WORKING")
            break
        elif key == ord('o') and recognized_person:  # Open the door manually with 'o' only if a person is recognized
            # print("TRIGGER OPEN")
            trigger_door_open()
        elif key == ord('c'):  # Close the door manually with 'c'
            # print("TRIGGER CLOSED")
            trigger_door_close()
            time.sleep(5)  # Prevent immediate reopening for 5 seconds

    camera.release()
    cv2.destroyAllWindows()
