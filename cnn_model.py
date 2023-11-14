import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import to_categorical
from keras import losses, optimizers, metrics

classes= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times', '=']
num_classes = len(classes)
input_shape = (45, 45, 1)

# 学習データのダウンロード
X_train = np.load('./traindata/data.npz')['arr_0']
X_test = np.load('./traindata/data.npz')['arr_1']
y_train = np.load('./traindata/data.npz')['arr_2']
y_test = np.load('./traindata/data.npz')['arr_3']

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# Build the model
model = Sequential()
model.add(InputLayer(input_shape=input_shape))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# model.summary()

batch_size = 64
epochs = 30

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate=0.002, decay=1e-6),
    metrics=["accuracy"]
)

model.fit(X_train, y_train, 
          batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save('./save_model/cnn_model.h5')