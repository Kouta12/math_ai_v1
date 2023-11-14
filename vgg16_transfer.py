import numpy as np
from tensorflow import keras
from keras import Sequential, Model
from keras.layers import *
from keras.utils import to_categorical
from keras import optimizers
from keras.applications import VGG16

classes= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times', '=']
num_classes = len(classes)
input_shape = (224, 224, 1)

# 学習データのダウンロード
X_train = np.load('./traindata/data_02.npz')['arr_0']
X_test = np.load('./traindata/data_02.npz')['arr_1']
y_train = np.load('./traindata/data_02.npz')['arr_2']
y_test = np.load('./traindata/data_02.npz')['arr_3']

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
# print('y_train shape:', y_train.shape)
# print('y_test shape:', y_test.shape)


def build_model():
    # 独自の入力層を追加
    input_tensor = X_train.shape[1:]
    input_model = Sequential()
    input_model.add(InputLayer(input_shape=input_tensor))
    input_model.add(Conv2D(3, (3,3), padding='same'))
    input_model.add(BatchNormalization())
    input_model.add(Activation('relu'))

    # VGG16のモデル構築
    model = VGG16(include_top=False, weights=None, input_tensor=input_model.output)
    
    # 出力層の追加
    x = Flatten()(model.layers[-1].output)
    x = Dense(num_classes, activation='softmax')(x)
    
    return Model(model.inputs, x) 

model = build_model()
model.summary()

opt = optimizers.Adam(learning_rate=0.001, decay=1e-6)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

# weightの読み込み
model.load_weights('./save_model/vgg16_weights.h5', by_name=True)

epochs = 30
iteration_train = 1500
batch_size_train = int(X_train.shape[0] / iteration_train)

model.fit(X_train, y_train, 
          batch_size=batch_size_train, epochs=epochs, validation_split=0.1)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save('./save_model/vgg16_transfer_02.h5')

