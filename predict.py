from keras.models import load_model
import imutils
from keras.preprocessing.image import img_to_array

import numpy as np
import cv2
import dataset_tool

def predict(path):
    print('[INFO] loading network...')
    model = load_model('class.model')

    image = cv2.imread(path)
    orig = image.copy()

    image = cv2.resize(image, (dataset_tool.image_size, dataset_tool.image_size))
    image = image.astype('float') / 255.0

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    result = model.predict(image)[0]
    proba = np.max(result)
    label = str(np.where(result == proba)[0])
    label = "{}: {:.2f}%".format(label, proba * 100)
    print(label)

    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Output', output)
    cv2.waitKey(0)

predict('02039_00002.png')

