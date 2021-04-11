import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense, Concatenate, Reshape, LSTM
from keras.models import Sequential, Model

import keras
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras import backend as K
import time
from sklearn.model_selection import StratifiedKFold

num_classes = 3
batch_size = 128
img_rows, img_cols, num_chan = 8, 9, 4
# img_rows, img_cols, num_chan = 17, 19, 5

falx = np.load('/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/t6x_89.npy')
y = np.load('/home/kaka/Desktop/sfy_file/eeg_emotion/nonCrossSubject/xnz/data/DE0.5s/t6y_89.npy')

# one_y = np.array([y[:1697]] * 3).reshape((-1,))
# one_y = to_categorical(one_y, num_classes)
one_y_1 = np.array([y[:1126]] * 3).reshape((-1,))
one_y_1 = to_categorical(one_y_1, num_classes)


acc_list = []
std_list = []
all_acc = []
for nb in range(15):
    K.clear_session()
    start = time.time()
    one_falx_1 = falx[nb * 3:nb * 3 + 3]
    one_falx_1 = one_falx_1.reshape((-1, 6, img_rows, img_cols, 5))

    # ###============= random select ============####
    # permutation = np.random.permutation(one_y_1.shape[0])
    # one_falx_2 = one_falx_1[permutation, :]
    # one_falx = one_falx_2[0:3400]
    # one_y_2 = one_y_1[permutation, :]
    # one_y = one_y_2[0:3400]
    # ###============= random select ============####
    one_y = one_y_1
    one_falx = one_falx_1[:,:,:,:,1:5]

    print(one_y.shape)
    print(one_falx.shape)
    # x_train, x_test, y_train, y_test = train_test_split(one_falx, one_y, test_size=0.25)
    seed = 7
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []


    # create model
    for train, test in kfold.split(one_falx, one_y.argmax(1)):
        img_size = (img_rows, img_cols, num_chan)


        def create_base_network(input_dim):

            seq = Sequential()
            seq.add(Conv2D(64, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
            seq.add(Conv2D(128, 4, activation='relu', padding='same', name='conv2'))
            seq.add(Conv2D(256, 4, activation='relu', padding='same', name='conv3'))
            seq.add(Conv2D(64, 1, activation='relu', padding='same', name='conv4'))
            seq.add(MaxPooling2D(2, 2, name='pool1'))
            seq.add(Flatten(name='fla1'))
            seq.add(Dense(512, activation='relu', name='dense1'))
            seq.add(Reshape((1, 512), name='reshape'))

            return seq




        base_network = create_base_network(img_size)
        input_1 = Input(shape=img_size)
        input_2 = Input(shape=img_size)
        input_3 = Input(shape=img_size)
        input_4 = Input(shape=img_size)
        input_5 = Input(shape=img_size)
        input_6 = Input(shape=img_size)



        out_all = Concatenate(axis=1)([base_network(input_1), base_network(input_2), base_network(input_3), base_network(input_4), base_network(input_5), base_network(input_6)])
        lstm_layer = LSTM(128, name='lstm')(out_all)
        out_layer = Dense(3, activation='softmax', name='out')(lstm_layer)
        model = Model([input_1, input_2, input_3, input_4, input_5, input_6], out_layer)
        # model.summary()

        # Compile model
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        # Fit the model
        x_train = one_falx[train]
        y_train = one_y[train]

        model.fit([x_train[:, 0], x_train[:, 1], x_train[:, 2], x_train[:, 3], x_train[:, 4], x_train[:, 5]], y_train, epochs=100, batch_size=128, verbose=0)
        # evaluate the model
        x_test = one_falx[test]
        y_test = one_y[test]
        scores = model.evaluate([x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3], x_test[:, 4], x_test[:, 5]], y_test, verbose=0)

        print("%.2f%%" % (scores[1] * 100)) # Accuracy
        all_acc.append(scores[1] * 100)

    # print("all acc: {}".format(all_acc))
    print("mean acc: {}".format(np.mean(all_acc)))
    print("std acc: {}".format(np.std(all_acc)))
    acc_list.append(np.mean(all_acc))
    std_list.append(np.std(all_acc))
    print("进度： {}".format(nb))
    all_acc = []
    end = time.time()
    print("%.2f" % (end - start))   # run time
print('Acc_all: {}'.format(acc_list))
print('Std_all: {}'.format(std_list))
print("Acc_mean: {}".format(np.mean(acc_list)))
print("Std_all: {}".format(np.std(std_list)))