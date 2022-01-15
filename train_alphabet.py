from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
# read data
# download: https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format
path = "C:/vs_data/handwritten/A_Z Handwritten Data.csv" # This is my csv path
df = pd.read_csv(path).astype('float32')
X = df.drop('0', axis = 1)
y = df['0']
# Because each row in the data containing 785 columns, which the first is
# result, the remaining columns is a 784 length vector.
# Reshaping the data csv file so that it can be displayed as an image 784 = 28x28
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
    20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
alphabets = []
for i in word_dict.values():
    alphabets.append(i)
# alphabet has 26 characters, so the output is one of 26 characters
# Reshaping the training and test dataset so that it can be put into the model
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train_one = to_categorical(y_train, 26, dtype = 'int')
y_test_one = to_categorical(y_test, 26, dtype = 'int')
# Define the model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'valid'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))

model.add(Dense(26, activation = 'softmax'))
# compiling and fitting model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train_one, epochs = 3, batch_size = 128, validation_data = (x_test, y_test_one))
model.save(r'model_alphabet.h5')
# Getting the train and validation accuracy & losses
print("The validation accuracy is: ", history.history['val_accuracy'])
print("The training accuracy is: ", history.history['accuracy'])
print("The validation loss is: ", history.history['val_loss'])
print("The training loss is: ", history.history['loss'])



