# # import cv2
# #
# # # Open the webcam
# # cap = cv2.VideoCapture(0)  # 0 for default camera
# #
# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     # Show the video frame
# #     cv2.imshow("Live Video", frame)
# #
# #     # Press 'q' to exit
# #     if cv2.waitKey(1) & 0xFF == ord("q"):
# #         break
# #
# # cap.release()
# # cv2.destroyAllWindows()
# #
# # # Convert to HSV color space
# # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# #
# # # Show the HSV image (for testing)
# # cv2.imshow("HSV Image", hsv_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# #
# # # Define color ranges in HSV (change values for different colors)
# # color_ranges = {
# #     "Red": ([0, 120, 70], [10, 255, 255]),
# #     "Green": ([40, 40, 40], [80, 255, 255]),
# #     "Blue": ([100, 150, 50], [140, 255, 255]),
# #     "Yellow": ([20, 100, 100], [30, 255, 255])
# # }
# #
# # lower_red = np.array([0, 120, 70])
# # upper_red = np.array([10, 255, 255])
# #
# # # Create a mask to filter red color
# # mask = cv2.inRange(hsv_image, lower_red, upper_red)
# #
# # # Show the mask
# # cv2.imshow("Red Mask", mask)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# #
# # # Apply mask to extract only the red color
# # red_result = cv2.bitwise_and(image, image, mask=mask)
# #
# # # Show the extracted red color
# # cv2.imshow("Red Color Detection", red_result)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
#
# # import cv2
# # import numpy as np
# #
# # # Function to detect color based on HSV value
# # def detect_color(h, s, v):
# #     if 0 <= h <= 10 or 170 <= h <= 180:
# #         return "Red"
# #     elif 35 <= h <= 85:
# #         return "Green"
# #     elif 100 <= h <= 140:
# #         return "Blue"
# #     elif 20 <= h <= 35:
# #         return "Yellow"
# #     elif 10 <= h <= 20:
# #         return "Orange"
# #     elif 140 <= h <= 170:
# #         return "Purple"
# #     else:
# #         return "Unknown"
# #
# # # Open webcam
# # cap = cv2.VideoCapture(0)
# #
# #
# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     cv2.imshow("Live Video", frame)
# #
# #     # Get frame dimensions
# #     height, width, _ = frame.shape
# #     center_x, center_y = width // 2, height // 2
# #
# #     # Convert frame to HSV color space
# #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# #
# #     # Get the HSV value of the center pixel
# #     pixel_center = hsv[center_y, center_x]
# #     h, s, v = int(pixel_center[0]), int(pixel_center[1]), int(pixel_center[2])
# #
# #     # Detect color
# #     color_name = detect_color(h, s, v)
# #
# #     # Print the detected color
# #     print(f"Detected Color: {color_name}")
# #
# #     # Press 'q' to exit
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # cap.release()
# # cv2.destroyAllWindows()
#
#
# import cv2
# import numpy as np
# import requests
# import time
#
# # ESP32 Server Details
# ESP32_IP = "http://10.73.177.163/"  # Replace with ESP32's actual IP
#
# # Color ranges in HSV (Adjust as needed)
# COLOR_RANGES = {
# #"red": ([0, 90, 50], [15, 255, 255]),
#     # "red": ([0, 120, 70], [10, 255, 255]),
#     #"red1": ([0, 150, 80], [10, 255, 255]),  # Lower Red (pure red shades)
#     #"red2": ([170, 150, 80], [180, 255, 255]) , # Upper Red (deep red shades)
#     "green": ([40, 70, 70], [80, 255, 255]),
#     # "blue": ([100, 150, 70], [140, 255, 255]),
#     # "yellow": ([20, 100, 100], [30, 255, 255]),
#     #"orange": ([10, 100, 100], [20, 255, 255]),
#     #"purple": ([130, 50, 50], [160, 255, 255]),
#     "pink": ([160, 100, 100], [180, 255, 255]),
#     #"cyan": ([85, 100, 100], [100, 255, 255]),
#    # "brown": ([10, 100, 20], [20, 255, 200]),
#     #"lime": ([30, 150, 100], [40, 255, 255]),
#     #"teal": ([80, 150, 100], [90, 255, 255]),
#     #"magenta": ([140, 50, 50], [170, 255, 255]),
#     #"silver": ([0, 0, 170], [180, 50, 255]),
#     #"gold": ([20, 150, 100], [40, 255, 255]),
#     #"navy":  ([100, 50, 50], [130, 255, 100]),
#     #"maroon": ([0, 100, 50], [10, 255, 150]),
#     #"olive": ([30, 50, 50], [50, 255, 150]),
#     #"beige": ([15, 30, 150], [25, 100, 255]),
#    # "black": ([0, 0, 0], [180, 255, 50])
# }
#
# # Capture video from camera
# cap = cv2.VideoCapture("http://192.168.0.197:4747/video")
#
# detected_colors = set()  # Keep track of already sent colors
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert frame to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     current_frame_colors = set()  # Track detected colors in the current frame
#
#     for color, (lower, upper) in COLOR_RANGES.items():
#         lower_bound = np.array(lower)
#         upper_bound = np.array(upper)
#
#         # Create mask for each color
#         mask = cv2.inRange(hsv, lower_bound, upper_bound)
#
#         # Apply morphological operations to remove noise & connect nearby regions
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Closes small gaps
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Removes small noise
#
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Use RETR_EXTERNAL
#
#         merged_contours = []
#         for contour in contours:
#             if cv2.contourArea(contour) > 500:  # Filter out small detections
#                 merged_contours.append(contour)
#
#         if merged_contours:
#             # Find the largest contour (assuming it's the main object)
#             largest_contour = max(merged_contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
#
#
#             # Draw rectangle around detected object
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#             # Compute center of object
#             obj_x = int(x + w / 2)
#             obj_y = int(y + h / 2)
#
#             print(f"Detected {color} object at: X={obj_x}, Y={obj_y}")
#
#             current_frame_colors.add(color)
#
#             # Send color to ESP32 if not already sent
#             if color not in detected_colors:
#                 try:
#                     response = requests.get(f"{ESP32_IP}/python?data={color}")
#                     print(f"Sent '{color}' to ESP32, Response: {response.text}")
#                 except requests.exceptions.RequestException as e:
#                     print(f"Error sending data to ESP32: {e}")
#
#     # Update detected colors set
#     detected_colors = current_frame_colors
#
#     # Show output frame
#     cv2.imshow('Object Detection', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
#

import cv2
import numpy as np
import requests
import threading

# ESP32 Server Details
ESP32_IP = "http://10.73.177.163/"  # Replace with ESP32's actual IP

# Color ranges in HSV
COLOR_RANGES = {
    "green": ([40, 70, 70], [80, 255, 255]),
    "pink": ([160, 100, 100], [180, 255, 255]),
    # "red1": ([0, 150, 80], [10, 255, 255]),  # Lower Red
    # "red2": ([170, 150, 80], [180, 255, 255]),  # Upper Red
     "blue": ([100, 150, 70], [140, 255, 255]),
    "yellow": ([20, 100, 100], [30, 255, 255])
}

# Capture video from phone camera
cap = cv2.VideoCapture("http://192.168.0.197:4747/video")

detected_color = None  # Last detected color
send_lock = threading.Lock()  # Prevent multiple requests at the same time

def send_color_to_esp32(color):
    """ Sends detected color to ESP32 without blocking video processing. """
    global detected_color
    with send_lock:
        if color != detected_color:
            try:
                response = requests.get(f"{ESP32_IP}/python?data={color}")
                print(f"Sent '{color}' to ESP32, Response: {response.text}")
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
            # Draw contours
           # cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)

            detected_in_frame = color  # Update detected color

    # If a color is detected, send it asynchronously
    if detected_in_frame:
        threading.Thread(target=send_color_to_esp32, args=(detected_in_frame,)).start()

    # Show output frame (video continues running)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
