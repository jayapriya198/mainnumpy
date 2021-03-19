#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Activation , Flatten , Dropout
from keras import backend as K
import numpy as np
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:


img_width, img_height=150,150
train_data_dir='F:/datasetgnu/training'
validation_data_dir='F:/datasetgnu/validation'
nb_train_samples=142
nb_validation_samples=142
epochs=5
batch_size=6

if K.image_data_format()=='channels_first':
    input_shape=(3,img_width,img_height)
else:
    input_shape=(img_width,img_height,3)
    
train_datagen=ImageDataGenerator(
    rescale=1. /255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1. /255)

train_generator=train_datagen.flow_from_directory(
        train_data_dir,
         target_size=(img_width,img_height),
         batch_size=batch_size,
         class_mode='binary')

validation_generator=test_datagen.flow_from_directory(
        validation_data_dir,
         target_size=(img_width,img_height),
         batch_size=batch_size,
         class_mode='binary')
    


# In[11]:


Model=Sequential()
Model.add(Conv2D(32,(3,3),input_shape=input_shape))
Model.add(Activation('relu'))
Model.add(MaxPool2D(pool_size=(2,2)))

Model.summary()

Model.add(Conv2D(32,(3,3)))
Model.add(Activation('relu'))
Model.add(MaxPool2D(pool_size=(2,2)))

Model.add(Conv2D(64,(3,3)))
Model.add(Activation('relu'))
Model.add(MaxPool2D(pool_size=(2,2)))

Model.add(Flatten())
Model.add(Dense(64))
Model.add(Activation('relu'))
Model.add(Dropout(0.5))
Model.add(Dense(1))
Model.add(Activation('sigmoid'))

Model.summary()

Model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

Model.fit_generator(
      train_generator,
      steps_per_epoch=nb_train_samples // batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=nb_validation_samples // batch_size)

Model.save_weights('first_try.h5')

img_pred=image.load_img('F:/datasetgnu/validation/hones/10.JPG', target_size=(150,150))
img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred,axis=0)


rslt=Model.predict(img_pred)
print(rslt)
if rslt[0][0]==0:
    prediction="hnot"
else:
    prediction="hone"
print(prediction)




# In[2]:


img =  validation_generator[130]
plt.imshow(validation_generator[130])
test_img = tf.reshape(img, [1,150,150,3])
img_class = model.predict_classes(test_img)
prediction = img_class[0]
classname = img_class[0]
print("Class: ",classname)


# In[ ]:


rslt=model.predict(img_pred)
print(rslt)
if rslt[0][0]==1:
    prediction="hnot"
else:
    prediction="hone"
print(prediction)



image = tf.keras.preprocessing.image.load_img(abspath('F:/datasetgnu/validation/hnot.2.JPG'))
input_arr = keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)

