# USAGE
# python execute_xray_pb.py --image dataset/COVID1.png 


"""
'input' = --image xray1.jpg 
'model' = model/model_generalize_val_29March.json , model/model_generalize_val_29March.h5

'predicted classes' = class_names[first]
                      class_names[second]
					  class_names[third]

'output heatmap' = output
"""

# import the necessary packages
import cv2
import argparse
import imutils
import numpy as np
from PIL import Image

#Tensorflow
import tensorflow as tf
from gradcam import GradCAM
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.preprocessing import image

#Preprocessing
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

model = tf.keras.models.load_model('graph/')
# Check its architecture
model.summary()

# load the original image from disk (in OpenCV format) and then
# resize the image to its target dimensions
orig = cv2.imread(args["image"])
resized = cv2.resize(orig, (224, 224))

# load the input image from disk (in Keras/TensorFlow format) and
# preprocess it
image = load_img(args["image"], target_size=(224, 224))
img = np.array(image)
img = resize(img, (224,224,3),anti_aliasing=True)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])

#Predict the classes
classes = model.predict(images, batch_size=1)
conf = classes[0]
top_3 = np.argsort(classes[0])[:-4:-1]
first = top_3[0]
second = top_3[1]
third = top_3[2]

class_names = ['COVID19','HEALTHY','PNEUMONIA','TB']
print(class_names[first],'=',conf[first]*100 ,'%')
print(class_names[second],'=',conf[second]*100,'%')
print(class_names[third],'=',conf[third]*100,'%')
print("Done ")
print(" ")


# initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, first)
heatmap = cam.compute_heatmap(images)

# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

# draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 120), (0, 0, 0), -1)
cv2.putText(output, f"{class_names[first]} = {(conf[first]*100):.2f} % ", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 255, 255), 2)
cv2.putText(output, f"{class_names[second]} = {(conf[second]*100):.2f} % ", (10, 85), cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 255, 255), 2)

# display the original image and resulting heatmap and output image
# to our screen
result = imutils.resize(output, height=700)
cv2.imwrite('result.png',result)
cv2.imshow("Output", result)
cv2.waitKey(0)


