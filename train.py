from ultralytics import YOLO

# Load a pretrained model
model = YOLO('runs/pose/yolo_pose_100e/weights/best.pt')

# Train the model
results = model.train(
    data='hand_landmark.yaml',
    epochs=100,
    imgsz=1280,
    batch=16,
    name='pose_2nd_100'
)

