from Yolov2 import Yolo_Utils, YoloV2
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# To make sure we are using tf2.0
print("TensorFlow version is ", tf.__version__)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------
# Main Model Parameters ------------------------------------------------------------
input_image_H, input_image_W = 416, 416
grid_sizes = (13, 13)
obj_score_threshold = 0.3
non_max_sup_threshold = 0.3

anchor_boxes = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
anchor_boxes_buffers = 20

# ----------------------------------------------------------------------------------
# Instantiating YOLO model      ----------------------------------------------------
yolo = YoloV2.Yolo((input_image_H, input_image_W, 3), grid_sizes, obj_score_threshold,
                   non_max_sup_threshold)

# Loading pre-trained weights
yolo.load_pretrained_weights()

# ----------------------------------------------------------------------------------
# Making Prediction on an IMAGE ----------------------------------------------------

test_image_path = 'test_image_4.jpg'
test_image = image.load_img(test_image_path, target_size=(416, 416))
test_image = image.img_to_array(test_image)
test_image /= 255.
x = np.expand_dims(test_image, axis=0)

# Making Prediction on an image
x_out = yolo.make_coded_prediction(x)

boxes = Yolo_Utils.decode_net_output(x_out[0], anchor_boxes,
                                     obj_score_threshold=obj_score_threshold,
                                     non_max_sup_threshold=non_max_sup_threshold)

plt.figure(figsize=(10, 10))
x = Yolo_Utils.draw_bboxes_image(test_image, boxes, obj_labels=yolo.obj_labels)

plt.imshow(x)
plt.show()

# ----------------------------------------------------------------------------------
# Making Prediction on a VIDEO  ----------------------------------------------------
input_video = 'MelbourneTraminFedSquareH264.mov'
output_video = 'YOLO_output_video.avi'

# Processing the video to make predictions on it
Yolo_Utils.draw_bboxes_video(yolo, input_video, output_video)

# Playing the output video file
Yolo_Utils.play_video(output_video)

# ----------------------------------------------------------------------------------
# Making Prediction on a Calgary Traffic Camera Outputs  ---------------------------

# obj_dict = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3}


image_save_folder = './camera_images'

# Camera Location: 12 Avenue / 5 Street SW
cam_url = 'http://trafficcam.calgary.ca/loc76.jpg'

# Analyzing the camera output
obj_counts_df = Yolo_Utils.detection_on_camera(yolo, cam_url, 60, 24, image_save_folder)

obj_counts_df.to_csv('recorded_identified_objects.csv', index=False)

# ----------------------------------------------------------------------------------
# Plotting the results -------------------------------------------------------------
results = pd.read_csv('recorded_identified_objects.csv', index_col=-1)
results.index = pd.to_datetime(results.index, format='%Y-%m-%d %H:%M')
results = results.drop(columns=results.columns[0])

# Plotting the results
Yolo_Utils.plot_results(results, cam_url)


