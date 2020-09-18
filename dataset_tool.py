
from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical
from imutils import paths
import numpy as np
import random
import cv2
import os

image_size = 32
CLASSES_NUM = 62

# 完整输出矩阵
np.set_printoptions(threshold=np.inf)

def load_data(path):
    data = []
    labels = []
    # 排序
    imagePaths = sorted(list(paths.list_images(path)))
    # 打乱排序，固定随机顺序
    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (image_size, image_size))

        image = img_to_array(image)
        data.append(image)

        label = int(imagePath.split(os.path.sep)[-2])
        labels.append(label)

    # 归一化
    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=CLASSES_NUM)
    return data, labels

# train_path = 'D:/BaiduNetdiskDownload/traffic-sign/train'
# load_data(train_path)