import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = x_test.reshape(x_test.shape[0], -1) / 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
# 定义优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    # 得到损失率
    metrics=['accuracy']
)
# (60000/32)=训练次数 然后再来100次循环
model.fit(x_train, y_train, batch_size=32, epochs=1)

loss, accuracy = model.evaluate(x_test, y_test)

print('loss:', loss, 'accuracy:', accuracy)

print(y_test[0])

x_test_element = x_test[0].reshape(-1, 784)

Y_pred = model.predict(x_test_element, batch_size=1)

print(Y_pred)
