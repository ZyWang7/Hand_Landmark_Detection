from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# define the pairs of keypoints need to connect together
skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
            [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
            [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]]


def plot_keypoints(image, keypoints, thickness):
    """function to plot the keypoints """
    for i in range(len(keypoints)):
        key = keypoints[i]
        for j in range(len(key)):
            x = int(key[j][0])
            y = int(key[j][1])
            image = cv2.circle(image, (x,y), radius=0, color=(0, 0, 255), thickness=thickness)
    return image


def plot_line(image, skeleton, keypoints, thickness):
    """ fuction to plot the lines between the keypoints """
    start_point = (225, 0)
    for i in range(len(keypoints)):
        key = keypoints[i]

        for sk in skeleton:
            start_idx = sk[0]
            end_idx = sk[1]
            start_point = (int(key[start_idx][0]), int(key[start_idx][1]))
            # print(f"start_point is: {start_point}")
            end_point = (int(key[end_idx][0]), int(key[end_idx][1]))
            # print(f"end_point is: {end_point}")

            image = cv2.line(image, start_point, end_point, color=(128, 255, 0), thickness=thickness)
    return image


def predict(model, image_path, saved_path):
    """ Predict the image and save the plolted image to the given path """
    results = model.predict(image_path)

    for r in results:
        im_array = r.plot(kpt_radius=0)  # plot a BGR numpy array of predictions
        image = np.array(im_array[..., ::-1])

        # get the coords of the keypoints
        keypoints = r.keypoints.xy.numpy()

        W = image.shape[0]
        thickness = W // 120
        # plot keypoints, skeleton
        line_image = plot_line(image, skeleton, keypoints, thickness-1)
        kpt_image = plot_keypoints(line_image, keypoints, thickness+2)
        # save image
        mpimg.imsave(saved_path, kpt_image)


# test
model = YOLO('runs/pose/pose_2nd_100/weights/best.pt')
predict(model, "Data/image/CSK6-003-004_000131.jpg", "test.jpg")
