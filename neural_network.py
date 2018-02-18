import os

import cv2
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def alex_net_model():
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='input', input_shape=(227, 227, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='max_pool1'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (5, 5), activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='max_pool2'))
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), activation='relu', name='conv3'))
    model.add(Conv2D(384, (3, 3), activation='relu', name='conv4'))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv5'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='max_pool3'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(4096))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(30))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def create_model(birds, types, batch_size, epochs):
    encoder = LabelEncoder()
    encoder.fit(types)
    encoded_y_train = encoder.transform(types)
    categorical_types = np_utils.to_categorical(encoded_y_train)
    model = alex_net_model()
    model.fit(birds, categorical_types, batch_size=batch_size, epochs=epochs)
    model.save_weights('trained_network.h5')
    return model


def load_model(path):
    model = alex_net_model()
    model.load_weights(path)
    return model


def load_images(path):
    birds = []
    types = []

    for bird_type in os.listdir(path):
        for pic in os.listdir(path + '//' + bird_type):
            bird = cv2.imread(path + '//' + bird_type + '//' + pic)
            birds.append(cv2.resize(bird, (227, 227)))
            types.append(bird_type)

    birds = np.asarray(birds)
    types = np.asarray(types)

    return birds, types


def predict(trained_network):
    model = load_model(trained_network)
    test_x, test_y = load_images('processed_images/TEST')
    encoder = LabelEncoder()
    encoder.fit(test_y)
    encoded_y_test = encoder.transform(test_y)
    test_y = np_utils.to_categorical(encoded_y_test)
    predicted = np.rint(model.predict(test_x))
    print('Accuracy: ', accuracy_score(test_y, predicted))
