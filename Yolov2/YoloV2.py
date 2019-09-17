import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization, Conv2D, Concatenate, Reshape, Dense, \
    Flatten, LeakyReLU, ReLU, MaxPool2D, Softmax

import numpy as np


class Yolo:
    """
    This class builds a Full YOLOv2 model with all the pre-trained weights loaded. This model
    detects 80 objects. The maximum number of bounding boxes per each grid is 5.
    It contains the following functions:
        --Predict:
    """

    def __init__(self, input_img_size=(416, 416, 3), grid_sizes=(13, 13), obj_score_threshold=0.3,
                 non_max_sup_threshold=0.3):
        self.obj_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                           'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                           'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                           'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                           'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                           'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                           'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.obj_classes = len(self.obj_labels)
        self.image_size = input_img_size
        self.grid_sizes = grid_sizes
        self.num_bboxes = 5
        self.anchor_boxes = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
        self.anchor_boxes_buffers = 50
        self.output_dim = self.num_bboxes * (1 + 4 + self.obj_classes)
        self.obj_score_threshold = obj_score_threshold
        self.non_max_sup_threshold = non_max_sup_threshold

        # Defining the layers of YOLOv2 using keras
        self.input_bbox = Input(shape=(1, 1, 1, self.anchor_boxes_buffers, 4))
        input_image = Input(shape=input_img_size)

        # Layer 1
        X = Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_1')(input_image)
        X = BatchNormalization(name='norm_1')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_1')(X)
        X = MaxPool2D(pool_size=(2, 2), name='max_pool_1')(X)

        # Layer 2
        X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_2')(X)
        X = BatchNormalization(name='norm_2')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_2')(X)
        X = MaxPool2D(pool_size=(2, 2), name='max_pool_2')(X)

        # Layer 3
        X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_3')(X)
        X = BatchNormalization(name='norm_3')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_3')(X)

        # Layer 4
        X = Conv2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Conv_4')(X)
        X = BatchNormalization(name='norm_4')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_4')(X)

        # Layer 5
        X = Conv2D(128, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_5')(X)
        X = BatchNormalization(name='norm_5')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_5')(X)

        X = MaxPool2D(pool_size=(2, 2), name='max_pool_5')(X)

        # Layer 6
        X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_6')(X)
        X = BatchNormalization(name='norm_6')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_6')(X)

        # Layer 7
        X = Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Conv_7')(X)
        X = BatchNormalization(name='norm_7')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_7')(X)

        # Layer 8
        X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_8')(X)
        X = BatchNormalization(name='norm_8')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_8')(X)

        X = MaxPool2D(pool_size=(2, 2), name='max_pool_8')(X)

        # Layer 9
        X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_9')(X)
        X = BatchNormalization(name='norm_9')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_9')(X)

        # Layer 10
        X = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Conv_10')(X)
        X = BatchNormalization(name='norm_10')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_10')(X)

        # Layer 11
        X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_11')(X)
        X = BatchNormalization(name='norm_11')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_11')(X)

        # Layer 12
        X = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Conv_12')(X)
        X = BatchNormalization(name='norm_12')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_12')(X)

        # Layer 13
        X = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_13')(X)
        X = BatchNormalization(name='norm_13')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_13')(X)

        # To get better idea about Skip-Connection (ResNets) check this source:
        # https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
        X_shortcut = X

        X = MaxPool2D(pool_size=(2, 2), name='max_pool_13')(X)

        # Layer 14
        X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_14')(X)
        X = BatchNormalization(name='norm_14')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_14')(X)

        # Layer 15
        X = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Conv_15')(X)
        X = BatchNormalization(name='norm_15')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_15')(X)

        # Layer 16
        X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_16')(X)
        X = BatchNormalization(name='norm_16')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_16')(X)

        # Layer 17
        X = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Conv_17')(X)
        X = BatchNormalization(name='norm_17')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_17')(X)

        # Layer 18
        X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_18')(X)
        X = BatchNormalization(name='norm_18')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_18')(X)

        # Layer 19
        X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_19')(X)
        X = BatchNormalization(name='norm_19')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_19')(X)

        # Layer 20
        X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_20')(X)
        X = BatchNormalization(name='norm_20')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_20')(X)

        # Layer 21 - Shourtcut Connection
        X_shortcut = Conv2D(64, (1, 1), strides=(1, 1), padding='same', use_bias=False, name='Conv_21_Shortcut')(
            X_shortcut)
        X_shortcut = BatchNormalization(name='norm_21_Shortcut')(X_shortcut)
        X_shortcut = LeakyReLU(alpha=0.1, name='leakR_21_Shortcut')(X_shortcut)
        X_shortcut = tf.nn.space_to_depth(X_shortcut, block_size=2, name='Reorg_21_Shortcut')

        X = Concatenate()([X_shortcut, X])

        # Layer 22
        X = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='Conv_22')(X)
        X = BatchNormalization(name='norm_22')(X)
        X = LeakyReLU(alpha=0.1, name='leakR_22')(X)

        # Layer 23
        # Note: In contrast to previous layers, Bias is not FALSE here.
        X = Conv2D(self.output_dim, (1, 1), strides=(1, 1), padding='same', name='Conv_23')(X)
        X_output = Reshape((grid_sizes[0], grid_sizes[1], self.num_bboxes, 1 + 4 + self.obj_classes))(X)

        # Construct the model
        self.model = Model(inputs=[input_image, self.input_bbox], outputs=[X_output])
        self.x_out = X_output

        # Print a summary of model
        self.model.summary()

    def load_pretrained_weights(self):
        # Loading pre-trained weights
        self.model.load_weights('./Yolov2/yolo.h5')
        print('PRE-TRAINED WEIGHTS ARE SUCCESSFULLY LOADED!')

    def make_coded_prediction(self, input_image):
        bbox_dummy = np.zeros((1, 1, 1, self.anchor_boxes_buffers, 4))
        bbox_dummy = np.expand_dims(bbox_dummy, axis=0)
        self.coded_output = self.model.predict([input_image, bbox_dummy])
        return self.coded_output
