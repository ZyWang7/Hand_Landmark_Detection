from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model

# Predict with the model
model.predict('bus.jpg', save=True)
