# Object_Detection

## Overview
In this project, I built a YOLOv2 model using the pre-trained weights to identify different objects in one of the City of Calgary traffic cameras to recognize the rush hours in terms of number of cars, buses, bikes, and pedestrains. YOLO (you only look once) is a real-time object detection system. You can find more details on it on the authors' website that can be found [here](https://pjreddie.com/darknet/yolov2/).  

To run the model given in this repository, you need to download the pre-trained weights of YOLOv2 ('yolo.h5') first. Several versions of weight files are available but the one I found the best can be found [here](https://drive.google.com/uc?id=11Q0Zq_bQSusPP8ALA3yeZq9j0yMfMBe-&export=download).  

Then, you need to either put this file in Yolov2 package under this repository or change the file_path regarding the weight file in 'load_pretrained_weights' function of YoloV2 module in Yolov2 package.

Using main.py file, you can make predictions (object detection) either on an Image, Video, or Images captured from traffic camera using camera's url.

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

