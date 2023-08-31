from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n-pose.pt')

# Train the model
results = model.train(
    data='hand_landmark.yaml',
    epochs=200,
    imgsz=1280,
    batch=16,
    name='pose_aug_200',
    flipud=0.5,
    fliplr=0.5
)

