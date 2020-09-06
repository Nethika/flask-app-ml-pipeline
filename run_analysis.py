import io
import os
import time
import math
import sys
#from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

import dlib
import glob
from imutils.face_utils import FaceAligner
import imutils
from imutils import paths
import numpy as np
from scipy.spatial import distance
import json
import string

from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.optimizers import SGD
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam

import cv2
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import emotion_model

import h5py
import scipy.io
import scipy.misc
import PIL
from PIL import Image
from PIL import ImageOps

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from yolo_video import *
from yolo import YOLO

import datetime

import random
import skimage.io


import coco
import utils
import model as modellib
import visualize



# Load Models

# Yolo3
#yolo = YOLO()
#graph_yolo3 = tf.get_default_graph()
#print("INFO..............Yolo-3 Loaded")

## Face landmarks model:
predictor_path = "models/shape_predictor_68_face_landmarks.dat" 
sp = dlib.shape_predictor(predictor_path)
## Face recognition model(# calculates 128D vector (hash) for an image):
face_rec_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
## model from dlib to detect faces:
detector = dlib.get_frontal_face_detector()
## to allign the face    
#fa = FaceAligner(sp, desiredFaceWidth=512)  
print("INFO..............Face Detection Model Loaded")

## Age and Gender model
model_a_g = load_model("models/gender_age.model")
model_a_g._make_predict_function()
print("INFO..............Age & Gender Model Loaded")


# Load Model for Emotion 
model_em = emotion_model.model_emotion()
#model.summary()
print("INFO..............Emotion Model Loaded")
emo_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
graph_em = tf.get_default_graph()

### Mask
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
mask_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
COCO_MODEL_PATH = "mask_rcnn_coco.h5"
mask_model.load_weights(COCO_MODEL_PATH, by_name=True)
graph_mask = tf.get_default_graph()

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

print("INFO..............Mask Model Loaded")

print("")
print("INFO..............Waiting for Images....")


class Watcher:
    DIRECTORY_TO_WATCH = "model_start"
    

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
            #print("Watcher for FaceID Stopped")

        self.observer.join()


class Handler(FileSystemEventHandler):
    @staticmethod
    def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    @staticmethod
    def on_any_event(event):
        global sp
        global facerec
        global detector
        global model_a_g


        
        if event.is_directory:
            return None

        #elif event.event_type == 'created' and event.src_path[-4:] == ".jpg":
        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            #print("Received created event - %s." % event.src_path)
            new_image_path = event.src_path
            file_name = os.path.basename(event.src_path)

            
            prof_macthed_wrote = False
            prof_macthed = False

            #calculate image_shape
            im_pil = Image.open(new_image_path)
            #print("INFO..............Image Loaded:",new_image_path)
            #im.save("test_cropped_1.jpg",quality=100)
            # returns (width, height) tuple
            (im_width, im_height)= im_pil.size
            image_shape = (float(im_height), float(im_width))

            ## MASK
            m_image = skimage.io.imread(new_image_path)
            # Run detection
            with graph_mask.as_default():
                results = mask_model.detect([m_image], verbose=1)

            # Visualize results
            r = results[0]

            output_image = "result_images/" + file_name
            base_path = "static/img/"
            out_path = base_path + output_image

            visualize.display_instances(out_path,m_image, r['rois'], r['masks'], r['class_ids'], 
                                        class_names, r['scores'])




            

            



##########################

            img = cv2.imread(new_image_path)

            # input_img: RGB  (input to face detector)
            face_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


            img_h, img_w, _ = np.shape(img)
            img_size_a_g = 64
            img_size_e = 100
            img_size_a_g = 64
            img_size_emo = 48

            # detect faces using dlib detector
            detected = detector(face_input, 1)
            #print("INFO..............Faces Detected:",len(detected))

            faces = np.empty((len(detected), img_size_a_g, img_size_a_g, 3))

            #csv
            prof_file = "hashes/profiles.csv"

            features = []
            emotion_class =[]

            if len(detected) > 0:
                
                for i, d in enumerate(detected):
                    # create face hash
                    shape_new = sp(face_input, d)     # face landmarks model
                    face_descriptor = facerec.compute_face_descriptor(face_input, shape_new)     # face recognition model
                    features.append(face_descriptor)

                
                    # For Gender and Age
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - 0.4 * w), 0)
                    yw1 = max(int(y1 - 0.4 * h), 0)
                    xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                    yw2 = min(int(y2 + 0.4 * h), img_h - 1)

                    faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size_a_g, img_size_a_g))

                    #print("INFO..............Calculating Emotion.....")
                    #face_only = img[y1:y2,x1:x2]
                    face_only = img[yw1:yw2,xw1:xw2]
                    face_gray = (cv2.cvtColor(face_only, cv2.COLOR_RGB2GRAY)*(1./255))
                    im = cv2.resize(face_gray, (img_size_emo, img_size_emo))
                    im = im.reshape((1,img_size_emo,img_size_emo,1))
                    #class_probs = model_em.predict(im)
                    with graph_em.as_default():
                        e_class = model_em.predict_classes(im, verbose=0)[0]
                    #print("emotion_class:",emo_classes[e_class])
                    emotion = emo_classes[e_class]
                    #print(emotion)

                    #print("Face #: ",i)
                    #print ("Emotion:",emotion)
                    emotion_class.append(emotion)



                features = np.array(features)     


                #print("INFO..............Calculating Age and Gender.....")
                #print(model_a_g.summary())
                # predict ages and genders of the detected faces
                results = model_a_g.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                #print("Predicted Ages:",predicted_ages)
                #print("Predicted Gender:",predicted_genders)


                # Profile data

                df = pd.read_csv(prof_file, delimiter=',')
                prof_data = eval(df.to_json(orient='records'))



                # Threshold set to identify different faces.  Might need tuning in case of siblings, sun glass pictures etc.  
                threshold=0.4

                #

                for i in range(len(features)): #faces in new image
                    inface_hash= features[i]
                    match_dict={}
                    prof_match = {}

                    #Profile Data
                    for j in range(len(prof_data)):

                        face_hash = json.loads(prof_data[j]['hash'])


                        dist = distance.euclidean(inface_hash,face_hash)
                        #print(face_id , face_freq, dist)
                        if dist < threshold:
                            prof_match[j] = dist

                    if prof_match:
                        
                        indx = min(prof_match, key=prof_match.get)

                        person_name = prof_data[indx]['name']
                        real_age = prof_data[indx]['age']
                        real_gender = prof_data[indx]['gender']
                        bio = prof_data[indx]['about']

                        images_folder = prof_data[indx]['images']

                        #get image names

                        #base_path = "static/img/"
                        #images_folder = "profile/nethika"

                        prof_path = base_path + images_folder

                        im_list = os.listdir(prof_path)
                        # choose random 3
                        ims = np.random.choice(im_list, 3, replace=False)
                        image_1 = images_folder+"/"+ims[0]
                        image_2 = images_folder+"/"+ims[1]
                        image_3 = images_folder+"/"+ims[2]

                        prof_macthed = True


                    else:     #new face
                        person_name = "Not Identified"
                        real_age = ""
                        real_gender = ""
                        bio = ""
                        image_1 = ""
                        image_2 = ""
                        image_3 = ""

                    # write Json

                    if not prof_macthed_wrote:
                        face_gender = "Female" if predicted_genders[i][0] > 0.5 else "Male"
                        face_age = int(predicted_ages[i])
                        face_emotion = emotion_class[i]

                        result_dict = { "result_image_path": output_image, 
                                        "Calculated Age": face_age, 
                                        "Calculated Gender": face_gender,
                                        "Calculated Emotion": face_emotion,
                                        "Person Identified": person_name, 
                                        "Real Age": real_age,
                                        "Real Gender": real_gender,
                                        "About" : bio,
                                        "image1" : image_1,
                                        "image2" : image_2,
                                        "image3" : image_3
                                        } 
                        
                        

                        #print("Writing Results to Json File...")
                        json_file = "results.json"                                    
                        #update json file
                        with open(json_file, 'w') as outfile:
                            json.dump(result_dict, outfile)
                        print("INFO..............Saved json for:",output_image)



                        if prof_macthed:
                            prof_macthed_wrote = True



        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print("Received modified event - %s." % event.src_path)

        elif event.event_type == 'deleted':
            # Taken any action here when a file is modified.
            print("Recieved deleted event - %s." % event.src_path)

        else:
            print("Event recieved: %s." % event.event_type)



if __name__ == '__main__':
    w = Watcher()
    w.run()


#########################################################################################
