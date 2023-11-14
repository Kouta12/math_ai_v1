# math_ai_v1

使用したデータセット
https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols

img_inversion.pyで画像を白黒反転
gen_data.pyで訓練データとテストデータに分割
cnn_model.pyで最初のモデル構築
vgg16_transfer.pyでvgg16のモデルを使い転移学習

```
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
```
