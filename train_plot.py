from ultralytics.models.yolo.pose import PoseTrainer

args = dict(
    model='yolov8n-pose.pt',
    data='hand_landmark.yaml',
    epochs=200,
    imgsz=1280,
    batch=16,
    name='hand_plot_200',
    flipud=0.5,
    fliplr=0.5
)

trainer = PoseTrainer(overrides=args)
trainer.train()

