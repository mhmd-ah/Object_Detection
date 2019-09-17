# Object_Detection

## Overview
In this project, I built a YOLOv2 model using the pre-trained weights to identify different objects in one of the City of Calgary traffic cameras to recognize the rush hours in terms of number of cars, buses, bikes, and pedestrains. YOLO (you only look once) is a real-time object detection system that has been amongst the most popular object detection algorithms. Three versions of YOLO has been released, and in this project I employed YOLOv2 which is capable of detecting 80 objects using a maximum of 5 anchor boxes per grid. You can find more details on [authors' website](https://pjreddie.com/darknet/yolov2/). All the publications regarding different versions of YOLO can be found [here](https://pjreddie.com/publications/). 

To run the model given in this repository, you need to first download the pre-trained weights of YOLOv2 ('yolo.h5'). Several versions of weight files are available but the one I found to be correct can be found [here](https://drive.google.com/uc?id=11Q0Zq_bQSusPP8ALA3yeZq9j0yMfMBe-&export=download).  

Then, you need to either put this file in *Yolov2* package under this repository or change the *file_path* regarding the weight file in *__load_pretrained_weights__* function of *YoloV2* module in *Yolov2* package.

Using __main.py__ file, you can make predictions (object detection) either on an Image, Video, or Images captured from traffic camera using camera's url.

## Install
This project requires Python 2.7 or higher with the following libraries installed:
  * [numpy](https://numpy.org/)
  * [pandas](https://pandas.pydata.org/)
  * [tensorflow (r2.0)](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf)
  * [keras](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf)
  * [cv2](https://pypi.org/project/opencv-python/)
  * [matplotlib](https://matplotlib.org/)
  * [urllib](https://docs.python.org/2/library/urllib.html)
  
## Files Needed To Be Downloaded:
  * [Pre-trained Weights of YoloV2](https://drive.google.com/uc?id=11Q0Zq_bQSusPP8ALA3yeZq9j0yMfMBe-&export=download)

