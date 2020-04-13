import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import models
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,GlobalAveragePooling2D, Dropout
from keras import layers
from keras import Model
import logging
from keras.applications.inception_v3 import InceptionV3



X_size = 75
Y_size = 75

base_model = InceptionV3(include_top = False,input_shape=(X_size,Y_size,3), classes=52)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(52, activation='softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
datagen = ImageDataGenerator(rescale= 1./255)

train_generator = datagen.flow_from_directory('./trafficSignsHW/trainFULL',
        target_size=(X_size,Y_size),
        batch_size = 32,
        class_mode='categorical')
model.compile(keras.optimizers.Adam(), 'categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=20, epochs=32)
model.save("my_model.h5")
