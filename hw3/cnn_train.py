import pandas as pd
import numpy as np
import sys
import pickle
from math import floor
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.utils import to_categorical

# load data
def normalize(x):
    x = x / 255.
    return x

train_path = sys.argv[1]
train = pd.read_csv(train_path)
## 下面這句太猛啦
X = np.array([row.split(" ") for row in train["feature"].tolist()], dtype=np.float32)
y = np.array(train["label"].tolist())

with open('encoder.pkl', 'wb') as f:
    pickle.dump(X, f)
    pickle.dump(y, f)

with open('encoder.pkl', 'rb') as f:
    X = pickle.load(f)
    y = pickle.load(f)

# shuffle and split
def shuffle_split_data(X, y, percent):
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    X_train = X[0:int(floor(percent*X.shape[0]))]
    y_train = y[0:int(floor(percent*X.shape[0]))]
    X_valid = X[int(floor(percent*X.shape[0])):]
    y_valid = y[int(floor(percent*X.shape[0])):]

    print(len(X_train), len(y_train), len(X_valid), len(y_valid))
    return X_train, y_train, X_valid, y_valid

## 70% for training, 30% for validation
train_X, train_y, valid_X, valid_y = shuffle_split_data(X, y, percent=0.7)

# change y to one-hot vector (use with categorical_crossentropy)
train_y = to_categorical(train_y, num_classes=7)
valid_y = to_categorical(valid_y, num_classes=7)

# normalize X
## https://www.zhihu.com/question/52684594 tells what -1 means
train_X = normalize(train_X.reshape(-1,48,48,1))
valid_X = normalize(valid_X.reshape(-1,48,48,1))

# 進行Image Data Augmentation以擴增數據量
## https://zhuanlan.zhihu.com/p/30197320
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=[0.8, 1.2],
    shear_range=0.2,
    horizontal_flip=True
)
print("train_X shape", train_X.shape)
print("valid_X shape", valid_X.shape)

# hyperparameter
batch_size = 128
epochs = 400
input_shape = (48,48,1)

# build model
## first hidden layer
## 以Sequential()開頭，後面再開始add
model = Sequential()
## "same"表示填充輸入以使輸出具有與原始輸入相同的長度
model.add(Conv2D(64, input_shape=input_shape, kernel_size=(5,5), padding='same', kernel_initializer='glorot_normal'))
## advanced activation要用add加上去
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
## 做完normalization後再進行池化
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
## 最後記得要dropout
model.add(Dropout(0.25))

## second hidden layer
## no need to specify input_shape
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

## third hidden layer
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.35))

## fourth hidden layer
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.35))

## flatten
model.add(Flatten())

## fully-connected NN
## first hidden layer
model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
## second hidden layer
model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
## output layer
model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

# 編譯：選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 打印出模型概述信息
print(model.summary())

# callbacks: History, EarlyStopping, ModelCheckpoint
## History紀錄了損失函數和其他指標的數值隨epoch變化的情況，
## 如果有驗證集的話，也包含了驗證集的這些指標變化情況。
his = History()
## early stopping to prevent overfitting
## patience：沒有進步的訓練輪數
## verbose：詳細信息模式
early_stop = EarlyStopping(monitor='val_acc', patience=7, verbose=1)
## 在每個batch之後保存模型
check_save = ModelCheckpoint(
    ## 05d表示五位數，.5f表示小數點後五位
    "model/model1-{epoch:05d}-{val_acc:.5f}.h5", 
    monitor='val_acc',
    save_best_only=True
)

callbacks = [his, early_stop, check_save]

# start training
## fit_generator比fit還節省內存
model.fit_generator(
    datagen.flow(train_X, train_y, batch_size=batch_size), 
    ## 一個epoch包含的步數（每一步是一個batch的數據送入）
    steps_per_epoch=5*len(train_X)//batch_size,
    validation_data=(valid_X, valid_y),
    epochs=epochs, 
    callbacks=callbacks
)

# save model
model.save("model/model1.h5")