# 詳見https://keras.io/zh/
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta

def build_model():

    '''
    #先定義好框架
    #第一步從input吃起
    '''
    
    input_img = Input(shape=(48, 48, 1))
    
    '''
    先來看一下keras document 的Conv2D
    keras.layers.Conv2D(filters, kernel_size, strides=(1,1), padding='valid', 
                        data_format=None, dilation_rate=(1,1), activation=None, 
                        ues_bias=True, kernel_initializer='glorot_uniform', 
                        bias_initializer='zeros', kernel_regularizer=None, 
                        bias_regularizer=None, activity_regularizer=None, 
                        kernel_constraint=None, bias_constraint=None)
    '''

    # 64個(5*5)filters
    # no padding不會在原有輸入的基礎上添加新的像素
    # 將input放在最後面括號內，也是一種置入"self"參數的方式
    block1 = Conv2D(64, (5,5), padding='valid', activation='relu')(input_img) 
    # 在邊界加上為0的padding，重新變回48*48
    # 表示輸入中維度的順序，channels_last對應輸入尺寸為(batch, height, width, channels)
    block1 = ZeroPadding2D(padding=(2,2), data_format='channels_last')(block1)
    # Maxpooling也可以設stride
    block1 = MaxPooling2D(pool_size=(5,5), strides=(2,2))(block1)


    block2 = Conv2D(64, (3,3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3,3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block3)
    block3 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3,3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3,3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation=('relu'))(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # 輸出整個模型的摘要資訊，包含簡單的結構表與參數統計
    model.summary()
    return model