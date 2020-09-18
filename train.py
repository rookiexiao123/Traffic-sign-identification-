from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import numpy as np
import dataset_tool

init_lr = 1e-3
EPOCHS = 50
bs = 32


import six
from keras.models import Model
from keras.layers import (
            Input,
            Activation,
            Dense,
            Flatten
            )

from keras.layers.convolutional import (
            Conv2D,
            MaxPooling2D,
            AveragePooling2D
            )

from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation('relu')(norm)

def _conv_bn_relu(**conv_params):
    filters = conv_params['filters']
    kernel_size = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', (1, 1))
    kernel_initializer = conv_params.setdefault('kernel_initializer', 'he_normal')
    padding = conv_params.setdefault('padding', 'same')
    kernel_regularizer = conv_params.setdefault('kernel_regularizer', l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(input)

        return _bn_relu(conv)

    return f

def _bn_relu_conv_(**conv_params):
    filters = conv_params['filters']
    kernel_size = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', (1, 1))
    kernel_initializer = conv_params.setdefault('kernel_initializer', 'he_normal')
    padding = conv_params.setdefault('padding', 'same')
    kernel_regularizer = conv_params.setdefault('kernel_regularizer', l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(activation)

    return f

def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS], kernel_size=(1, 1), strides=(stride_width, stride_height), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides, is_first_block_of_first_layer=(is_first_layer and i == 0))(input)

        return input
    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=init_strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv_(filters=filters, kernel_size=(3, 3), strides=init_strides)(input)

        residual = _bn_relu_conv_(filters=filters, kernel_size=(3, 3), strides=init_strides)(input)

        return _shortcut(input, residual)

    return f

def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=init_strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input)

        else:
            conv_1_1 = _bn_relu_conv_(filters=filters, kernel_size=(1, 1), strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv_(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv_(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f

def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception('Input shape should be a tuple (nb_channels, nb_rows, nb_cols)')
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        block = _bn_relu(block)

        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]), strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)

        dense = Dense(units=num_outputs, kernel_initializer='he_normal', activation='softmax')(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])

class Triffic_Net:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))
        model.summary()

        return model

def train(aug, trainX, trainY, testX, testY):
    print('[INFO] compiling model...')
    model = Triffic_Net.build(width=dataset_tool.image_size, height=dataset_tool.image_size, depth=3, classes=dataset_tool.CLASSES_NUM)
    #model = ResnetBuilder.build_resnet_152((3, dataset_tool.image_size, dataset_tool.image_size), dataset_tool.CLASSES_NUM)
    opt = Adam(lr=init_lr, decay=init_lr / EPOCHS)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print('[INFO] training work...')

    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=bs), validation_data=(testX, testY), steps_per_epoch=len(trainX) // bs, epochs=EPOCHS, verbose=1)

    print('[INFO] serializing network...')

    model.save('class.model')

    plt.style.use('ggplot')
    plt.figure()
    N = EPOCHS

    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
    plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')

    plt.title("Training loss and accuracy on traffic-sign classifier")
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')


if __name__ == '__main__':
    train_path = 'D:/BaiduNetdiskDownload/traffic-sign/train'
    test_path = 'D:/BaiduNetdiskDownload/traffic-sign/test'

    trainx, trainy = dataset_tool.load_data(train_path)
    testx, testy = dataset_tool.load_data(test_path)

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    train(aug, trainx, trainy, testx, testy)

