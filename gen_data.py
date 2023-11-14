import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

classes= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times', '=']
num_classes = len(classes)
image_size = 224

X = []
Y = []


# 画像を数値化しラベル分け {"+":10, "-":11, "times":12, "=":13}
for index, classlabel in enumerate(classes):
    photos_dir = './images/' + classlabel
    files = glob.glob(photos_dir + '/*.jpg')
    for i, file in enumerate(files):
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                    test_size=0.1,
                                                    shuffle=True,
                                                    stratify=Y)

# x = (X_train, X_test)
# y = (y_train, y_test)

## csvで保存
np.savez('./traindata/data_02', X_train, X_test, y_train, y_test)
