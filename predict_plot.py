from ultralytics.utils import ASSETS
from ultralytics.models.yolo.pose import PosePredictor

args = dict(model='runs/pose/hand_plot_200/weights/best.pt', source="custom_imgs/", save=True, conf=0.5)
predictor = PosePredictor(overrides=args)
predictor.predict_cli()

