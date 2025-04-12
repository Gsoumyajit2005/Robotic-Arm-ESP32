import cv2
import numpy as np
import requests
import threading

# ESP32 Server Details
ESP32_IP = "http://192.168.0.213/"  # Replace with ESP32's actual IP

# Color ranges in HSV
COLOR_RANGES = {
    "green": ([40, 70, 70], [80, 255, 255]),
    "pink": ([160, 100, 100], [180, 255, 255]),
    # "red1": ([0, 150, 80], [10, 255, 255]),  # Lower Red
    # "red2": ([170, 150, 80], [180, 255, 255]),  # Upper Red
    # "blue": ([100, 150, 70], [140, 255, 255]),
    # "yellow": ([20, 100, 100], [30, 255, 255])
}

# Capture video from phone camera
cap = cv2.VideoCapture("http://192.168.0.215:4747/video")

detected_color = None  # Last detected color
send_lock = threading.Lock()  # Prevent multiple requests at the same time

def send_color_to_esp32(color, x, y):
    """ Sends detected color and position (x, y) to ESP32 without blocking video processing. """
    global detected_color
    with send_lock:
        if color != detected_color:
            try:
                url = f"{ESP32_IP}/python?data={color}&x={x}&y={y}"
                response = requests.get(url)
                print(f"Sent '{color}' with position (X={x}, Y={y}) to ESP32. Response: {response.text}")
                detected_color = color  # Update last sent color
            except requests.exceptions.RequestException as e:
                print(f"Error sending data to ESP32: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_in_frame = None  # Track detected color in current frame

    for color, (lower, upper) in COLOR_RANGES.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Morphological operations for noise reduction
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  # Remove small objects

        if filtered_contours:
            largest_contour = max(filtered_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Draw rectangle around detected object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Compute center of object
            obj_x = int(x + w / 2)
            obj_y = int(y + h / 2)

            print(f"Detected {color} object at: X={obj_x}, Y={obj_y}")

            detected_in_frame = color  # Update detected color

            # If a color is detected, send it asynchronously with object position
            threading.Thread(target=send_color_to_esp32, args=(detected_in_frame, obj_x, obj_y)).start()

    # Show output frame (video continues running)
    cv2.imshow('Object Detection', frame)
    frame = cv2.resize(frame, (600, 420))  # Or smaller

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
