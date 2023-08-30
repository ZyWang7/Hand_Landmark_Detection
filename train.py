from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n-pose.pt')

# Train the model
results = model.train(
    data='hand_landmark.yaml',
    epochs=100,
    imgsz=320,
    batch=16,
    name='train_test'
)

