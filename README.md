# Hand_Landmark_Detection
Predict the bounding boxes around the hand in a given image and detect 21 key-points on the hand, based on yolov8n-pose

- the data set is from iFYTEK Developer Competition: https://challenge.xfyun.cn/topic/info?type=hand-key-point
    - there are 982 images
    - they are take using 3 different devices


### 21 keypoints:

![Alt text](https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png "keypoints")


### About the files:
- `hand_landmark.yaml`
    - set the keypoints, number of class and the path of the dataset
- `load_data.ipynb`
    - extract the labels from the .json file and store into .txt files
    - split the dataset into training and testing sets -> "data/train", "data/test"
- `train.py`
    - train the model with pretrained weights 'yolov8n-pose.pt'
- `predict.py`
    - random predict 50 images from the test dataset
    - the line between keypoints (skeleton) is plot manually using opencv
- `predict.ipynb`
    - visualize the predict images
- folder `"ultralytics"` - https://github.com/ultralytics/ultralytics
    - `requirements.txt`, `setup.cfg`, `setup.py`
        - setup -> requirements (library) needed for Ultralytics
    - I modified **skeleton structure for keypoints** in `"ultralytics/utils/plotting.py"` in order to automatically plot the line between the keypoints while making predictions

- `train_plot.py`
    - train using **PoseTrainer** in ultralytics
- `prediction_plot.py`
    - predict custom images and videos using **PosePredictor** in ultralytics
    - this time, its predictions can automatically draw lines between keypoints (skeleton)
    - the results are stored in "runs/detect/train/"
