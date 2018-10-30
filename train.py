from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import keras.optimizers as opt
from keras.optimizers import SGD
import cv2
import numpy as np

nb_classes = 41
nb_digits = 11
nb_colors = 5
epochs = 100
train_size = 14842
val_size = 1653
batch_size = 32

classes = {0: "0-K", 1: "0-Y", 2: "0-M", 3: "0-C",
           4: "1-K", 5: "1-Y", 6: "1-M", 7: "1-C",
           8: "2-K", 9: "2-Y", 10: "2-M", 11: "2-C",
           12: "3-K", 13: "4-Y", 14: "5-M", 15: "6-C",
           16: "4-K", 17: "4-Y", 18: "4-M", 19: "4-C",
           20: "5-K", 21: "5-Y", 22: "5-M", 23: "5-C",
           24: "6-K", 25: "6-Y", 26: "6-M", 27: "6-C",
           28: "7-K", 29: "7-Y", 30: "7-M", 31: "7-C",
           32: "8-K", 33: "8-Y", 34: "8-M", 35: "8-C",
           36: "9-K", 37: "9-Y", 38: "9-M", 39: "9-C",
           40: "nodigit"}

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'nodigit']
colors = ['K', 'Y', 'M', 'C', 'nocolor']


def multiclasses_getter(x, y):
    size = y.shape[0]
    digit_one_hot = np.zeros((size, len(digits)), dtype=np.float32)
    color_one_hot = np.zeros((size, len(colors)))
    labels = np.argmax(y, axis=1)
    labels = [classes[i] for i in labels]
    digit_labels = [label.split('-')[0] if label is not "nodigit" else label for label in labels]
    color_labels = [label.split('-')[1] if label is not "nodigit" else 'nocolor' for label in labels]
    for i, digit_label in enumerate(digit_labels):
        digit_one_hot[i, digits.index(digit_label)] = 1.0
    for i, color_label in enumerate(color_labels):
        color_one_hot[i, colors.index(color_label)] = 1.0
    return [digit_one_hot, color_one_hot]


def multiclass_flow_from_directory(flow_from_directory_gen, multiclasses_getter):
    for x, y in flow_from_directory_gen:
        yield x, multiclasses_getter(x, y)


kernel = np.ones((2, 2), np.uint8)


def dilate(img):
    img_opencv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_dilate = cv2.dilate(img_opencv, kernel=kernel, iterations=2)
    return cv2.cvtColor(img_dilate, cv2.COLOR_BGR2RGB)


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


# SGD

inputs = Input(shape=(28, 28, 3))
x1 = Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same', kernel_initializer='random_uniform')(
    inputs)
x2 = Conv2D(filters=32, kernel_size=(15, 15), activation='relu', padding='same', kernel_initializer='random_uniform')(
    inputs)
x3 = Conv2D(filters=32, kernel_size=(23, 23), activation='relu', padding='same', kernel_initializer='random_uniform')(
    inputs)
x = Concatenate(axis=3)([x1, x2, x3])
x = MaxPooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.2)(x)
x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', kernel_initializer='random_uniform')(x)
x1 = Conv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same', kernel_initializer='random_uniform')(x1)
x2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', kernel_initializer='random_uniform')(x)
x2 = Conv2D(filters=64, kernel_size=(14, 14), activation='relu', padding='same', kernel_initializer='random_uniform')(
    x2)
x = Concatenate(axis=3)([x1, x2])
x = MaxPooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.2)(x)
x1 = Conv2D(filters=96, kernel_size=(1, 1), padding='same', kernel_initializer='random_uniform')(x)
x1 = Conv2D(filters=96, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x1)
x2 = Conv2D(filters=96, kernel_size=(1, 1), padding='same', kernel_initializer='random_uniform')(x)
x2 = Conv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same', kernel_initializer='random_uniform')(x2)
x = Concatenate(axis=3)([x1, x2])
x = MaxPooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.2)(x)
x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x_digits = Dense(units=1024, activation='relu', kernel_initializer='random_uniform')(x)
x_digits = Dropout(0.2)(x_digits)
x_colors = Dense(units=1024, activation='relu', kernel_initializer='random_uniform')(x)
x_colors = Dropout(0.2)(x_colors)
digit_predictions = Dense(nb_digits, activation='softmax', name="Digits")(x_digits)
color_predictions = Dense(nb_colors, activation='softmax', name="Colors")(x_colors)
# model = MobileNetV2(weights=None, input_tensor=inputs, classes=nb_classes)
model = Model(input=inputs, output=[digit_predictions, color_predictions])
model.summary()
adam = opt.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
learning_rate = 0.001
decay_rate = learning_rate / epochs
momentum = 0.8
opt = SGD(lr=learning_rate, decay=decay_rate, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

train_datagen = ImageDataGenerator(preprocessing_function=brightness_adjustment, data_format="channels_last",
                                   rotation_range=10, shear_range=0.2, zoom_range=0.2, fill_mode='nearest',
                                   rescale=1. / 255)
val_datagen = ImageDataGenerator(preprocessing_function=dilate, data_format="channels_last", rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(directory='../train_data', target_size=(28, 28),
                                                    batch_size=batch_size, class_mode="categorical", shuffle=True)

val_generator = train_datagen.flow_from_directory(directory='../val_data', target_size=(28, 28),
                                                  batch_size=batch_size, class_mode="categorical")
train_generator = multiclass_flow_from_directory(train_generator, multiclasses_getter)
val_generator = multiclass_flow_from_directory(val_generator, multiclasses_getter)
model.fit_generator(train_generator, validation_data=val_generator, validation_steps=val_size // batch_size,
                    epochs=epochs, steps_per_epoch=train_size // batch_size)

model.save("model/first_try.h5")
