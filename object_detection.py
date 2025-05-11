
# import cv2
# import torch
# import numpy as np
# import requests
# import threading
# from torchvision import models, transforms
# from torch import nn
# from PIL import Image
#
# # ============================
# # 🔧 Config
# # ============================
# ESP32_IP = "http://192.168.0.113/"  # Change to your ESP32 IP
# VIDEO_FEED_URL = "http://192.168.0.197:4747/video"  # Or 0 for webcam
# MODEL_PATH = "efficientnet_epoch9_download.pth"
# NUM_CLASSES = 4
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CLASS_NAMES = ['hazardous', 'non-recyclable', 'organic', 'recyclable']
#
# # ============================
# # 📦 Load Model
# # ============================
# def load_model():
#     model = models.efficientnet_b0(pretrained=False)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()
#     return model
#
# model = load_model()
#
# # Image transform (same as training/validation)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# # ============================
# # 🧠 Predict Class
# # ============================
# def classify_frame(frame):
#     pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         output = model(img_tensor)
#         _, pred = torch.max(output, 1)
#     return CLASS_NAMES[pred.item()]
#
# # ============================
# # 📡 Send to ESP32
# # ============================
# last_sent = None
# send_lock = threading.Lock()
#
# def send_prediction(pred, x, y):
#     global last_sent
#     with send_lock:
#         if pred != last_sent:
#             try:
#                 url = f"{ESP32_IP}/python?data={pred}&x={x}&y={y}"
#                 response = requests.get(url)
#                 print(f"Sent to ESP32: {pred} at ({x}, {y}) → {response.text}")
#                 last_sent = pred
#             except requests.RequestException as e:
#                 print(f"Error sending data: {e}")
#
# # ============================
# # 🎥 Video Stream + Prediction
# # ============================
# cap = cv2.VideoCapture(VIDEO_FEED_URL)  # Or 0 for webcam
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Could not read frame")
#         break
#
#     height, width, _ = frame.shape
#
#     # Get center region (or use full frame if needed)
#     center_crop = frame[height//4:3*height//4, width//4:3*width//4]
#     pred_class = classify_frame(center_crop)
#
#     # Compute center for position
#     obj_x, obj_y = width // 2, height // 2
#
#     # Draw label
#     cv2.putText(frame, f"{pred_class}", (obj_x - 50, obj_y - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#     cv2.circle(frame, (obj_x, obj_y), 8, (0, 255, 0), -1)
#
#     # Send to ESP32
#     threading.Thread(target=send_prediction, args=(pred_class, obj_x, obj_y)).start()
#
#     cv2.imshow("Waste Classifier", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import torch
# import numpy as np
# from torchvision import models, transforms
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from PIL import Image
#
# # ========= Setup =========
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CLASS_NAMES = ["hazardous", "non-recyclable", "organic", "recyclable"]
#
# # Load model
# model = models.efficientnet_b0(pretrained=False)
# in_features = model.classifier[1].in_features
# model.classifier[1] = torch.nn.Linear(in_features, 4)
# model.load_state_dict(torch.load("efficientnet_epoch9_download.pth", map_location=DEVICE))
# model.to(DEVICE)
# model.eval()
#
# # GradCAM setup
# target_layers = [model.features[-1]]
# cam = GradCAM(model=model, target_layers=target_layers)
#
# # Transform
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# # ========= Live Feed =========
# cap = cv2.VideoCapture("http://192.168.0.197:4747/video")  # Or use 0 for built-in webcam
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert to RGB and PIL Image
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(img_rgb)
#
#     # Apply transform
#     input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
#
#     # Get Grad-CAM
#     with torch.no_grad():
#         output = model(input_tensor)
#         pred_class = torch.argmax(output).item()
#         pred_label = CLASS_NAMES[pred_class]
#
#     # GradCAM heatmap
#     targets = [ClassifierOutputTarget(pred_class)]
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
#     img_np = np.array(pil_image).astype(np.float32) / 255.0
#     # Resize cam to match the original image
#     grayscale_cam_resized = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
#
#     # Normalize original image to [0,1] for show_cam_on_image
#     img_np = img_rgb.astype(np.float32) / 255.0
#
#     # Overlay Grad-CAM
#     cam_overlay = show_cam_on_image(img_np, grayscale_cam_resized, use_rgb=True)
#
#     # Resize to original frame size
#     cam_overlay_resized = cv2.resize(cam_overlay, (frame.shape[1], frame.shape[0]))
#
#     # Add predicted label
#     cv2.putText(cam_overlay_resized, f"Prediction: {pred_label}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#
#     # Display
#     cv2.imshow("Live Grad-CAM", cam_overlay_resized)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

####CODE TO BE USED FOR NOW:#####


# from ultralytics import YOLO
# from torchvision import models, transforms
# from torch import nn
# import torch
# import cv2
# from PIL import Image
# import numpy as np
# import threading
# import requests
# import time
#
# # ===== CONFIG =====
# ESP32_IP = "http://192.168.0.213/"
# VIDEO_FEED_URL = "http://192.168.1.115:4747/video"  # Or your IP cam feed
# MODEL_PATH = "best_model.pth"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Full class list, model is trained with all 4
# FULL_CLASS_NAMES = ['hazardous', 'non-recyclable', 'organic', 'recyclable']
#
# # Only allow these:
# ALLOWED_CLASSES = {1: 'non-recyclable', 3: 'recyclable'}
#
#
# class VideoStream:
#     def __init__(self, src=0):
#         self.cap = cv2.VideoCapture(src)
#         self.ret, self.frame = self.cap.read()
#         self.stopped = False
#         threading.Thread(target=self.update, daemon=True).start()
#
#     def update(self):
#         while not self.stopped:
#             if self.cap.isOpened():
#                 self.ret, self.frame = self.cap.read()
#             time.sleep(0.01)  # slight sleep to reduce CPU usage
#
#     def read(self):
#         return self.ret, self.frame
#
#     def stop(self):
#         self.stopped = True
#         self.cap.release()
#
# # ===== Load Classifier =====
# def load_classifier():
#     model = models.efficientnet_b0(pretrained=False)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(in_features, len(FULL_CLASS_NAMES))
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()
#     return model
#
# classifier_model = load_classifier()
#
# # Transform for classifier input
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# # ===== Load YOLOv8 Model =====
# yolo_model = YOLO("yolov8n.pt")
#
# # ===== ESP32 Send =====
# last_sent = None
# send_lock = threading.Lock()
#
# def send_prediction(pred, x, y):
#     global last_sent
#     with send_lock:
#         if pred != last_sent:
#             try:
#                 url = f"{ESP32_IP}/python?data={pred}&x={x}&y={y}"
#                 response = requests.get(url)
#                 print(f"Sent to ESP32: {pred} at ({x}, {y}) → {response.text}")
#                 last_sent = pred
#             except requests.RequestException as e:
#                 print(f"Error sending data: {e}")
#
# # ===== Classify Crop =====
# def classify_crop(crop):
#     pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#     img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         output = classifier_model(img_tensor)
#         _, pred = torch.max(output, 1)
#         pred_class = pred.item()
#         if pred_class in ALLOWED_CLASSES:
#             return ALLOWED_CLASSES[pred_class]
#         else:
#             return None  # Skip this
#
# # ===== Main Loop =====
# stream = VideoStream(VIDEO_FEED_URL)
#
# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     results = yolo_model(frame)[0]
# #
# #     for box in results.boxes:
# #         x1, y1, x2, y2 = map(int, box.xyxy[0])
# #         conf = float(box.conf.item())
# #
# #         # Crop object
# #         cropped = frame[y1:y2, x1:x2]
# #         if cropped.size == 0:
# #             continue
# #
# #         # Classify
# #         waste_type = classify_crop(cropped)
# #         if waste_type is None:
# #             continue  # Skip if not recyclable or non-recyclable
# #
# #         # Draw box and label
# #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #         cv2.putText(frame, f"{waste_type} ({conf:.2f})", (x1, y1 - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
# #
# #         # Send to ESP32
# #         obj_x = (x1 + x2) // 2
# #         obj_y = (y1 + y2) // 2
# #         threading.Thread(target=send_prediction, args=(waste_type, obj_x, obj_y)).start()
# #
# #     cv2.imshow("YOLO + Filtered Classifier", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
#
# while True:
#     ret, frame = stream.read()
#     if not ret:
#         break
#
#     results = yolo_model(frame)[0]
#     boxes = results.boxes
#
#     if boxes is not None and len(boxes) > 0:
#         # Sort by confidence (descending)
#         boxes = sorted(boxes, key=lambda b: b.conf.item(), reverse=True)
#         top_box = boxes[0]  # Pick only the most confident box
#
#         x1, y1, x2, y2 = map(int, top_box.xyxy[0])
#         conf = float(top_box.conf.item())
#
#         # Crop object
#         cropped = frame[y1:y2, x1:x2]
#         if cropped.size == 0:
#             continue
#
#         # Classify
#         waste_type = classify_crop(cropped)
#         if waste_type is None:
#             continue  # Skip if not recyclable or non-recyclable
#
#         # Draw box and label
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{waste_type} ({conf:.2f})", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#         # Send to ESP32
#         obj_x = (x1 + x2) // 2
#         obj_y = (y1 + y2) // 2
#         threading.Thread(target=send_prediction, args=(waste_type, obj_x, obj_y)).start()
#
#     cv2.imshow("YOLO + Filtered Classifier (Single Object)", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# stream.stop()
# cv2.destroyAllWindows()
#

# from ultralytics import YOLO
# from torchvision import models, transforms
# from torch import nn
# import torch
# import cv2
# from PIL import Image
# import numpy as np
# import threading
# import requests
# import time
#
# # ===== CONFIG =====
# ESP32_IP = "http://192.168.0.213/"
# VIDEO_FEED_URL = "http://192.168.1.197:4747/video"  # Or your IP cam feed
# MODEL_PATH = "best_model.pth"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Only 2 classes now
# FULL_CLASS_NAMES = ['organic', 'recyclable']
# ALLOWED_CLASSES = {0: 'non-recyclable.', 1: 'recyclable'}  # Assuming 0 = organic, 1 = recyclable in your trained model
#
# class VideoStream:
#     def __init__(self, src=0):
#         self.cap = cv2.VideoCapture(src)
#         self.ret, self.frame = self.cap.read()
#         self.stopped = False
#         threading.Thread(target=self.update, daemon=True).start()
#
#     def update(self):
#         while not self.stopped:
#             if self.cap.isOpened():
#                 self.ret, self.frame = self.cap.read()
#             time.sleep(0.01)
#
#     def read(self):
#         return self.ret, self.frame
#
#     def stop(self):
#         self.stopped = True
#         self.cap.release()
#
# # ===== Load Classifier =====
# def load_classifier():
#     model = models.efficientnet_b0(pretrained=False)
#     in_features = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(in_features, len(FULL_CLASS_NAMES))  # len = 2
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#     model.to(DEVICE)
#     model.eval()
#     return model
#
#
# classifier_model = load_classifier()
#
# # Transform for classifier input
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# # ===== Load YOLOv8 Model =====
# yolo_model = YOLO("yolov8n.pt")
#
# # ===== ESP32 Send =====
# last_sent = None
# send_lock = threading.Lock()
#
# def send_prediction(pred, x, y):
#     global last_sent
#     with send_lock:
#         if pred != last_sent:
#             try:
#                 url = f"{ESP32_IP}/python?data={pred}&x={x}&y={y}"
#                 response = requests.get(url)
#                 print(f"Sent to ESP32: {pred} at ({x}, {y}) → {response.text}")
#                 last_sent = pred
#             except requests.RequestException as e:
#                 print(f"Error sending data: {e}")
#
# # ===== Classify Crop =====
# def classify_crop(crop):
#     pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#     img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         output = classifier_model(img_tensor)
#         _, pred = torch.max(output, 1)
#         pred_class = pred.item()
#         return ALLOWED_CLASSES.get(pred_class, None)
#
# # ===== Main Loop =====
# stream = VideoStream(VIDEO_FEED_URL)
#
# while True:
#     ret, frame = stream.read()
#     if not ret:
#         break
#
#     results = yolo_model(frame)[0]
#     boxes = results.boxes
#
#     if boxes is not None and len(boxes) > 0:
#         boxes = sorted(boxes, key=lambda b: b.conf.item(), reverse=True)
#         top_box = boxes[0]
#
#         x1, y1, x2, y2 = map(int, top_box.xyxy[0])
#         conf = float(top_box.conf.item())
#
#         cropped = frame[y1:y2, x1:x2]
#         if cropped.size == 0:
#             continue
#
#         waste_type = classify_crop(cropped)
#         if waste_type is None:
#             continue
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{waste_type} ({conf:.2f})", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#
#         obj_x = (x1 + x2) // 2
#         obj_y = (y1 + y2) // 2
#         threading.Thread(target=send_prediction, args=(waste_type, obj_x, obj_y)).start()
#
#     cv2.imshow("YOLO + Filtered Classifier (Single Object)", frame)
#     if cv2.waitKey(1) & 0xFF == ord(' '):
#         break
#
# stream.stop()
# cv2.destroyAllWindows()


from ultralytics import YOLO
from torchvision import models, transforms
from torch import nn
import torch
import cv2
from PIL import Image
import numpy as np
import threading
import requests
import time
import torch.nn.functional as F


# ===== CONFIG =====
ESP32_IP = "http://192.168.0.213/"
VIDEO_FEED_URL = "http://192.168.1.197:4747/video"  # Or your IP cam feed
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Only 2 classes now
FULL_CLASS_NAMES = ['organic', 'recyclable']
#ALLOWED_CLASSES = {0: 'non-recyclable', 1: 'recyclable'}  # Assuming 0 = organic, 1 = recyclable in your trained model
SECOND_CLASS_NAMES = ['organic', 'recyclable', 'hazardous', 'e-waste']  # Adjust based on your actual training

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ===== Load Classifier =====
def load_classifier():
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(FULL_CLASS_NAMES))  # len = 2
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

SECOND_MODEL_PATH = "efficientnet_epoch9_download.pth"


def load_second_classifier():
    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(SECOND_CLASS_NAMES))

    model.load_state_dict(torch.load(SECOND_MODEL_PATH, map_location=DEVICE))  # Might still error here
    model.to(DEVICE)
    model.eval()

    return model  # ✅ This line is required!

try:
    second_classifier_model = load_second_classifier()
    print("Second model loaded successfully.")
except Exception as e:
    print(f"Failed to load second classifier: {e}")
    second_classifier_model = None


ALLOWED_CLASSES = {
    0: 'non-recyclable',
    1: 'recyclable'
}

SECOND_ALLOWED_CLASSES = {
    0: 'non-recyclable',
    1: 'recyclable',
    #2: 'hazardous',
   # 3: 'e-waste'
}



second_classifier_model = load_second_classifier()

classifier_model = load_classifier()

# Transform for classifier input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== Load YOLOv8 Model =====
yolo_model = YOLO("yolov8n.pt")

# ===== ESP32 Send =====
last_sent = None
send_lock = threading.Lock()

def send_prediction(pred, x, y):
    global last_sent
    with send_lock:
        if pred != last_sent:
            try:
                url = f"{ESP32_IP}/python?data={pred}&x={x}&y={y}"
                response = requests.get(url)
                print(f"Sent to ESP32: {pred} at ({x}, {y}) → {response.text}")
                last_sent = pred
            except requests.RequestException as e:
                print(f"Error sending data: {e}")

# ===== Classify Crop =====
def classify_crop(crop):
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # First model (Model A)
        output1 = F.softmax(classifier_model(img_tensor), dim=1)
        conf1, pred1 = torch.max(output1, 1)
        label1 = ALLOWED_CLASSES.get(pred1.item(), None)

        # Second model (Model B)
        output2 = F.softmax(second_classifier_model(img_tensor), dim=1)
        conf2, pred2 = torch.max(output2, 1)
        label2 = SECOND_ALLOWED_CLASSES.get(pred2.item(), None)

        print(f"Model A: {label1} ({conf1.item():.2f}) | Model B: {label2} ({conf2.item():.2f})")

        # Routing logic:
        if label2 in ['hazardous', 'e-waste']:
            return label2
        elif label2 == 'recyclable':
            return label2  # Prefer model B's recyclable
        elif label1 == 'organic':
            return label1  # Prefer model A's organic
        else:
            # fallback — e.g., low confidence or ambiguous
            return label2 or label1



# ===== Main Loop =====
stream = VideoStream(VIDEO_FEED_URL)

while True:
    ret, frame = stream.read()
    if not ret:
        break

    results = yolo_model(frame)[0]
    boxes = results.boxes

    if boxes is not None and len(boxes) > 0:
        boxes = sorted(boxes, key=lambda b: b.conf.item(), reverse=True)
        top_box = boxes[0]

        x1, y1, x2, y2 = map(int, top_box.xyxy[0])
        conf = float(top_box.conf.item())

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        waste_type = classify_crop(cropped)
        if waste_type is None:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{waste_type} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        obj_x = (x1 + x2) // 2
        obj_y = (y1 + y2) // 2
        threading.Thread(target=send_prediction, args=(waste_type, obj_x, obj_y)).start()

    cv2.imshow("YOLO + Filtered Classifier (Single Object)", frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

stream.stop()
cv2.destroyAllWindows()
