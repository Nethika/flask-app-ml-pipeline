import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
import numpy as np
import os

num_classes = 7 
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def model_emotion():
    #------------------------------
    #construct CNN structure
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    #------------------------------

    #------------------------------

    model.compile(loss='categorical_crossentropy'
        , optimizer=keras.optimizers.Adam()
        , metrics=['accuracy']
    )

    #------------------------------


    #model.load_weights('models/facial_expression_model_weights.h5') #load weights
    #root_path = "C:\\Users\\NethikaSuraweera\\Documents\\TinMan_Git\\neural-networks\\camera_age_gender_ethnicity_emotions_pipeline"
    #root_path = os.getcwd()
    #model.load_weights(root_path + "/models/facial_expression_model_weights.h5")

    model.load_weights("models/facial_expression_model_weights.h5")

    return model

def predict_emotion_path(image_path,model):
    #print(model.summary())
    #new_image_path = "images_small/UPMHASHHOUSELV220160811114751a.jpg"



    img = image.load_img(image_path, grayscale=True, target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255

    #emotion_prob = model.predict(x)
    #print("emotion_prob:",emotion_prob)
    e_class = model.predict_classes(x, verbose=0)[0]
    #print("emotion_class:",classes[e_class])
    emotion = classes[e_class]
    return emotion


"""
def predict_emotion_path(im,model):
    face_only = img[y1:y2,x1:x2]
    face_gray = (cv2.cvtColor(face_only, cv2.COLOR_RGB2GRAY)*(1./255))
    im = cv2.resize(face_gray, (img_size_e, img_size_e))
    im = im.reshape((1,img_size_e,img_size_e,1))
    class_labels = model.predict(im)
    
    print("Face #: ",i)
    print ("Emotion:",classes[class_labels])
    ethnicity_class.append(classes[class_labels])

    return custom
"""

if __name__ == "__main__":
    em_model = model_emotion()
    image_path = "images_small/face.jpg"
    predict_emotion_path(image_path,em_model)