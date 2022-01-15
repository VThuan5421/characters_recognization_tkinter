from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
# To load data with sklearn
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version = 1)
data = mnist['data'].values
target = mnist['target'].values
# Reshape a vector to an array 2D so that it's can be displayed as an image
data = np.asarray(data).reshape((data.shape[0], 28, 28))
# Split dataset to training set and testset
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 7)
"""
# To load data with keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def show_image(data, target):
    plt.rcParams['figure.figsize'] = (9, 9)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        num = np.random.randint(0, len(data))
        plt.imshow(data[num].reshape(28, 28), cmap = 'gray', interpolation = None)
        plt.title("Class: {}".format(target[num]))

    plt.tight_layout()
    plt.show()

#show_image(x_train, y_train)

# Normalizing data to binary data
x_train = x_train / 255.0
x_train = x_train.astype('float32')

x_test = x_test / 255.0
x_test = x_test.astype('float32')
# Because output is a value belong to 0 - 9 range. We use to_categorical
# to group the class result. Example: The output is 3 => result is
# [0. 0. 0. 1. 0. ... 0.]. This is one-hot vector
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# with CNN model, we adding one more dimesion to the training set
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))
# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax'))
# compiling and fitting model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, batch_size = 128, epochs = 10, verbose = 1, validation_data = (x_test, y_test))
print("The model has successfully trained.")
# Evaluate the model
score = model.evaluate(x_test, y_test, verbose = 0)
print("Test loss: ", score[0])
print("Accuracy: ", score[1])
# Save model with pickle
model.save("model.h5")
print("Saving the model as model.h5")

