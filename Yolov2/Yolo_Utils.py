""""
This module contains the following functions to help with decoding outputs of YOLO model:
        -- sigmoid_func
        -- softmax_func
        -- bbox_iou_calculator
        -- decode_net_output
        -- draw_bounding_boxes
        -- draw_bboxes_video
        -- play_video
        -- object_counter
        -- plot_results
"""
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import urllib.request
from datetime import datetime
import time


def sigmoid_func(x):
    return 1. / (1. + np.exp(-x))


def softmax_func(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def bbox_iou_calculator(bboxes_1, bboxes_2):
    """
    This function calculates the intersect over union (IOU) ratio of two bounding boxes.

    :param bboxes_1:  The x, y, width, and height of the first bounding box.
    :param bboxes_2: The x, y, width, and height of the second bounding box.
    :return: It returns the intersect over union (IOU) ratio of the two bounding boxes.
    """
    b1_x, b1_y, b1_w, b1_h = bboxes_1
    b2_x, b2_y, b2_w, b2_h = bboxes_2

    # Calculating the x, y values of top and bottom of each box
    b1_x1 = b1_x - b1_w / 2.
    b1_x2 = b1_x + b1_w / 2.
    b1_y1 = b1_y - b1_h / 2.
    b1_y2 = b1_y + b1_h / 2.

    b2_x1 = b2_x - b2_w / 2.
    b2_x2 = b2_x + b2_w / 2.
    b2_y1 = b2_y - b2_h / 2.
    b2_y2 = b2_y + b2_h / 2.

    # Finding the x, y coordinates of the intersection area
    x1_iou = max([b1_x1, b2_x1])
    x2_iou = min([b1_x2, b2_x2])
    y1_iou = max([b1_y1, b2_y1])
    y2_iou = min([b1_y2, b2_y2])

    # Calculating intersection Area
    intersect_area = max(x2_iou - x1_iou, 0) * max(y2_iou - y1_iou, 0)
    union_area = b1_w * b1_h + b2_w * b2_h - intersect_area

    iou = 0
    if union_area > 0:
        iou = intersect_area / union_area

    return iou


def decode_net_output(net_output, anchor_boxes, obj_score_threshold=0.35, non_max_sup_threshold=0.35):
    """
    This function decodes outputs of YOLO model to find the final bounding boxes and the objects they
    have embodied.

    :param net_output: This is the output of YOLO model having the shape of
         (number of grids along height, number of grids along width, number of bounding boxes, 5 + number of objects)
    :param anchor_boxes: This is a list containing the dimensions (width, height) of anchor boxes.
    :param obj_score_threshold: This is a threshold for the score of an object above which object is kept.
    :param non_max_sup_threshold: This is a threshold for IOU above which the bounding box is removed.
    :return: It returns the final list of bounding boxes that contains (x, y, width, height, confidence of object,
            index of detected object in the object label list) per bounding box.
    """
    grid_h, grid_w, num_bboxes = net_output.shape[:3]

    num_obj = net_output.shape[-1] - 5

    # Applying sigmoid on Pc's, as the classification is Binary.
    # There is either an Object (1) in the box or No Object (0.0).
    net_output[..., 4] = sigmoid_func(net_output[..., 4])

    # Applying softmax on Class Probabilities to make their summations equal to 1.0
    net_output[..., 5:] = net_output[..., 4][..., np.newaxis] * softmax_func(net_output[..., 5:])
    net_output[..., 5:] *= (net_output[..., 5:] > obj_score_threshold)

    # Finding the boxes with probabilities higher than object score threshold
    # Note: The x and y coordinates of boxes will be unified with respect to the input image
    # Default: Left-top corner of image is (0,0); Right-top corner of image is (1,0)
    final_bboxes = []
    obj_confidences = []

    for grid_i in range(grid_h):
        for grid_j in range(grid_w):
            for box in range(num_bboxes):
                box_data = net_output[grid_i, grid_j, box, :]
                b_x, b_y, b_w, b_h = box_data[:4]

                if np.sum(box_data[5:]) > 0:
                    obj_index = np.argmax(box_data[5:])
                    obj_confidence = box_data[5 + obj_index]
                    obj_confidences.append(obj_confidence)

                    # Calculating w, and h of bounding box
                    b_w = np.exp(b_w) * anchor_boxes[2 * box] / grid_w
                    b_h = np.exp(b_h) * anchor_boxes[2 * box + 1] / grid_h

                    # Calculating x, and y coordinates for the center of bounding box
                    b_x = (sigmoid_func(b_x) + grid_j) / grid_w
                    b_y = (sigmoid_func(b_y) + grid_i) / grid_h

                    # Appending the Unified box specifications
                    final_bboxes.append([b_x, b_y, b_w, b_h, obj_confidence, obj_index])

    # Sorting the boxes based on their confidences (descending)
    final_bboxes = [x for _, x in sorted(zip(obj_confidences, final_bboxes), reverse=True)]

    # Applying Non-Max Suppression to filter out the boxes with high IOU's (Intersection over Union)
    for i in range(len(final_bboxes)):
        for j in range(i + 1, len(final_bboxes)):

            if len(final_bboxes[j]) > 0 and len(final_bboxes[i]) > 0:
                iou = bbox_iou_calculator(final_bboxes[i][:4], final_bboxes[j][:4])

                if iou > non_max_sup_threshold:
                    # Removing the box which has high IoU from the list
                    final_bboxes[j] = []

    # Removing de-selected boxes
    final_bboxes = [box for box in final_bboxes if len(box) > 0]

    return final_bboxes


def draw_bboxes_image(image, bboxes, obj_labels):
    """
    This function draws the final bounding boxes on the image, and returns the resulting image.

    :param image: This is the input image with the shape of (416, 416, 3).
    :param bboxes: This is a list containing the information of the final boxes (x, y, width, height,
                   confidence of object, index of detected object in the object label list).
    :param obj_labels: This is a list containing the label of all the objects which can be detected by YOLO model.
    :return: It returns an image of shape (416, 416, 3) with bounding boxes on it.
    """
    image_h, image_w, _ = image.shape

    for box in bboxes:
        # Calculating the x , y coordinate of the box
        x, y, w, h, score, obj_index = box

        # Note that the coordinates in CV2 should be Integer
        x1 = int((x - w / 2.) * image_w)
        x2 = int((x + w / 2.) * image_w)
        y1 = int((y - h / 2.) * image_h)
        y2 = int((y + h / 2.) * image_h)

        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

        cv2.putText(image,
                    (obj_labels[obj_index] + ':' + str(score)),
                    (x1, y1 + 10),
                    cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1e-3 * image_h,
                    color=(0, 255, 0), thickness=1)

    return image


def draw_bboxes_video(yolo_object, input_video, output_video):
    """
    This function draws the final bounding boxes on a video, and returns the resulting video.

    :param yolo_object: This is the instantiated YOLO object.
    :param input_video: This is the input video.
    :param output_video: This the input video with bounding boxes on it.
    :return: It returns the resulting output video.
    """
    video_reader = cv2.VideoCapture(input_video)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video_reader.get(cv2.CAP_PROP_FPS)
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Note: The frame size of output video should be the same as those of input video,
    #       otherwise, the writer will not write any image in the output video.
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (frame_w, frame_h))

    for i in tqdm(range(nb_frames)):
        # frame captured from video
        read_flag, cap_frame = video_reader.read()

        if read_flag:
            frame = cv2.resize(cap_frame, (416, 416))
            frame = image.img_to_array(frame)
            frame /= 255.
            x = np.expand_dims(frame, axis=0)

            x_out = yolo_object.make_coded_prediction(x)

            boxes = decode_net_output(x_out[0], yolo_object.anchor_boxes,
                                      obj_score_threshold=yolo_object.obj_score_threshold,
                                      non_max_sup_threshold=yolo_object.non_max_sup_threshold)

            frame = draw_bboxes_image(cap_frame, boxes, obj_labels=yolo_object.obj_labels)

            video_writer.write(frame)

    print("Video Processing is COMPLETE!")

    video_reader.release()
    video_writer.release()


def play_video(video):
    """
    This function plays the given MP4 video.
    :param video: This is a MP4 video.
    :return: NONE
    """
    # Create a VideoCapture object and read from input file
    cap_video = cv2.VideoCapture(video)

    while cap_video.isOpened():
        # Capture frame-by-frame
        read_flag, frame = cap_video.read()
        if read_flag:
            # Resizing the window on which video plays
            cv2.namedWindow('Frame')
            cv2.resizeWindow('Frame', 800, 600)

            # Resizing the frame to that of window
            frame = cv2.resize(frame, (800, 600))

            # Display the resulting frame
            cv2.imshow('Frame', frame)

        # Defining a key (that is the Q/q key) to EXIT the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the captured video
    cap_video.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def object_counter(final_boxes):
    """
    This function counts the objects
    {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'bus':5}
    detected on the image using the output of YOLO model.
    :param final_boxes: It is the list of bounding boxes that contains (x, y, width, height, confidence of object,
            index of detected object in the object label list).
    :return: A list (length of 5) containing the number of detected objects.
    """
    # obj_dict = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'bus':5}
    counts = [0] * 5
    for box in final_boxes:
        if box[-2] > 0:
            obj = box[-1]
            if obj < 4:
                counts[obj] += 1
            if obj == 5:
                counts[-1] += 1

    return counts


def plot_results(data_df, url):
    """
    This function plots the results of the objects detected on the
    images captured from a traffic Camera at given url.
    :param data_df: Dataframe containing the object detection results.
    :param url: Url of the traffic camera.
    :return: NONE
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in range(data_df.shape[1]):
        ax.scatter(data_df.index, data_df.iloc[:, col], label=data_df.columns[col])
        ax.plot(data_df.index, data_df.iloc[:, col])
    ax.legend(loc='upper right')
    ax.set_ylabel('Number of Objects')
    ax.set_xlabel('Date/Time')
    ax.set_ylim(np.min(np.array(data_df.values)), np.max(np.array(data_df.values)) + 1)
    ax.set_xlim(data_df.index[0], data_df.index[-1])
    ax.set_title('Camera Location: ' + url)
    plt.savefig('detected_objs.jpg')
    plt.show()


def detection_on_camera(yolo, cam_url, image_update_time, time_window, image_save_folder):
    num_images_to_analyse = int(time_window * 3600 / image_update_time) + 1
    image_shape = (yolo.image_size[0], yolo.image_size[1])

    columns = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'time']
    obj_counts_df = pd.DataFrame(columns=columns)

    for i in range(num_images_to_analyse):
        # Reading the camera image
        # try/except block is used to prevent "HTTP Error 404" error
        try:
            with urllib.request.urlopen(cam_url) as url:
                print(f'Reading Image   {i}   from Camera\n')

                # Recording time
                time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Reading/Saving the image
                cam_image = url.read()
                cam_image = np.asarray(bytearray(cam_image), dtype="uint8")
                cam_image = cv2.imdecode(cam_image, cv2.IMREAD_COLOR)
                f_name = "/img" + str(i) + ".jpg"
                cv2.imwrite(image_save_folder + f_name, cam_image)
                cam_image = cv2.resize(cam_image, image_shape)
                cam_image = cam_image / 255.
                cam_image = np.expand_dims(cam_image, axis=0)

                x_out = yolo.make_coded_prediction(cam_image)

                boxes = decode_net_output(x_out[0], yolo.anchor_boxes,
                                          obj_score_threshold=yolo.obj_score_threshold,
                                          non_max_sup_threshold=yolo.non_max_sup_threshold)

                obj_counts = object_counter(boxes)

                obj_counts.append(time_now)

                obj_counts_dict = dict(zip(columns, obj_counts))

                obj_counts_df = obj_counts_df.append(obj_counts_dict, ignore_index=True)

                if i % 10 == 0:
                    obj_counts_df.to_csv('recorded_identified_objects.csv')
        except:
            print('HTTP Error 404: The url could not be accessed!')

        time.sleep(image_update_time)
    return obj_counts_df
