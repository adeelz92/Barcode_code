from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import keras.optimizers as opt
from keras.optimizers import SGD
import cv2
import numpy as np
from src.models import create_model

nb_classes = 41
nb_digits = 11
nb_colors = 5
epochs = 100
train_size = 33173
val_size = 3415
batch_size = 64

classes = {0: "0-K", 1: "0-Y", 2: "0-M", 3: "0-C",
           4: "1-K", 5: "1-Y", 6: "1-M", 7: "1-C",
           8: "2-K", 9: "2-Y", 10: "2-M", 11: "2-C",
           12: "3-K", 13: "3-Y", 14: "3-M", 15: "3-C",
           16: "4-K", 17: "4-Y", 18: "4-M", 19: "4-C",
           20: "5-K", 21: "5-Y", 22: "5-M", 23: "5-C",
           24: "6-K", 25: "6-Y", 26: "6-M", 27: "6-C",
           28: "7-K", 29: "7-Y", 30: "7-M", 31: "7-C",
           32: "8-K", 33: "8-Y", 34: "8-M", 35: "8-C",
           36: "9-K", 37: "9-Y", 38: "9-M", 39: "9-C",
           40: "nodigit"}

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'nodigit']
colors = ['K', 'Y', 'M', 'C', 'nocolor']


def multiclasses_getter(y):
    size = y.shape[0]
    digit_one_hot = np.zeros((size, len(digits)), dtype=np.float32)
    color_one_hot = np.zeros((size, len(colors)), dtype=np.float32)
    labels = np.argmax(y, axis=1)
    labels = [classes[i] for i in labels]
    digit_labels = [label.split('-')[0] if label is not "nodigit" else label for label in labels]
    color_labels = [label.split('-')[1] if label is not "nodigit" else 'nocolor' for label in labels]
    for i, digit_label in enumerate(digit_labels):
        digit_one_hot[i, digits.index(digit_label)] = 1.0
    for i, color_label in enumerate(color_labels):
        color_one_hot[i, colors.index(color_label)] = 1.0
    return digit_one_hot


def multiclass_flow_from_directory(flow_from_directory_gen, multiclasses_getter):
    for x, y in flow_from_directory_gen:
        # x = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x]
        x = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in x]
        x = [cv2.dilate(img, kernel=kernel, iterations=2) for img in x]
        x = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in x]
        x = np.asarray([np.divide(img, 255) for img in x], dtype=np.float32)
        # x = np.asarray([np.expand_dims(img, axis=2) for img in x])
        yield x, multiclasses_getter(y)

kernel = np.ones((2, 2), np.uint8)

def dilate(img):
    img_opencv = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_dilate = cv2.dilate(img_opencv, kernel=kernel, iterations=2)
    return np.expand_dims(cv2.dilate(img_opencv, kernel=kernel, iterations=2), axis=2)


def brightness_adjustment(img):
    img_opencv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_dilate = cv2.dilate(img_opencv, kernel=kernel, iterations=2)
    img = cv2.cvtColor(img_dilate, cv2.COLOR_BGR2RGB)
    # turn the image into the HSV space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # creates a random bright
    ratio = .5 + np.random.uniform()
    # convert to int32, so you don't get uint8 overflow
    # multiply the HSV Value channel by the ratio
    # clips the result between 0 and 255
    # convert again to uint8
    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.int32) * ratio, 0, 255).astype(np.uint8)
    # return the image int the BGR color space
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

model = create_model()
#model = Model(input=inputs, output=[digit_predictions, color_predictions])
model.summary()
learning_rate = 0.001
decay_rate = learning_rate / epochs
momentum = 0.8
adam = opt.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_rate)
opt = SGD(lr=learning_rate, decay=decay_rate, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

train_datagen = ImageDataGenerator(data_format="channels_last", width_shift_range=0.2, height_shift_range=0.2)
val_datagen = ImageDataGenerator(data_format="channels_last")

train_generator = train_datagen.flow_from_directory(directory='../train_data', target_size=(28, 28),
                                                    batch_size=batch_size, class_mode="categorical", shuffle=True)

val_generator = val_datagen.flow_from_directory(directory='../val_data', target_size=(28, 28),
                                                batch_size=batch_size, class_mode="categorical", shuffle=True)
train_generator = multiclass_flow_from_directory(train_generator, multiclasses_getter)
val_generator = multiclass_flow_from_directory(val_generator, multiclasses_getter)
filepath = "model/first_try.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit_generator(train_generator, validation_data=val_generator, validation_steps=val_size // batch_size,
                    epochs=epochs, steps_per_epoch=train_size // batch_size, callbacks=[checkpoint])

# model.save("model/first_try.h5")
