import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense,Flatten, Conv2D
from keras.layers import BatchNormalization, MaxPooling2D, Dropout  # BatchNormalization 추가
from keras.utils import np_utils
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import TensorBoard


def keras_model(image_x, image_y):
    num_of_classes = 20         # 학습 데이터의 개수 증가 +5
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(image_x,image_y,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))   # strides (1, 1)로 변경
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))   # strides (1, 1)로 변경
    model.add(BatchNormalization())     # BatchNormalization 실행

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))     # Dropout 확률 0.6 -> 0.5로 변경
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))     # Dropout 확률 0.6 -> 0.5로 변경
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "QuickDrawNew.h5"    # 새롭게 학습한 QuickDrawNew 모델 파일 생성
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels


def prepress_labels(labels):
    labels = np_utils.to_categorical(labels)
    return labels


def main():
    features, labels = loadFromPickle()
    # features, labels = augmentData(features, labels)
    features, labels = shuffle(features, labels)
    labels=prepress_labels(labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.1)
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    model, callbacks_list = keras_model(28,28)
    model.summary()
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=20, batch_size=64,
              callbacks=[TensorBoard(log_dir="QuickDrawNew")])  # 새로운 QuickDrawNew로 log를 찍음
              # epochs는 3 -> 20으로 증가시켜 학습 횟수를 늘려서 정확도를 향상시킴
    model.save('QuickDrawNew.h5')   # 새롭게 작성한 모델 파일에 저장


main()
