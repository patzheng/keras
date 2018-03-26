import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = x_test.reshape(x_test.shape[0], -1) / 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential([
    Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax')
])
# 定义优化器
sgd = SGD(lr=0.1)

model.compile(
    optimizer=sgd,
    loss='mse',
    # 得到损失率
    metrics=['accuracy']
)
# (60000/32)=训练次数 然后再来100次循环
model.fit(x_train, y_train, batch_size=32, epochs=30)

loss, accuracy = model.evaluate(x_test, y_test)

print('loss:', loss, 'accuracy:', accuracy)
