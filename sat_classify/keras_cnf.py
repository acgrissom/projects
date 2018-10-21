from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os
import keras
from keras.layers import Conv1D, Conv2D, Flatten, SimpleRNN, Reshape, LSTM, GRU, Dropout

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

X = []
Y = []
NUM = 1000
pos_num = 0
neg_num = 0
EXAMPLES_DIR='sat_files'
NUM_CLAUSES = 220
NUM_VARIABLES = 50

shape = (NUM_CLAUSES, NUM_VARIABLES + 1)
for filename in os.listdir(EXAMPLES_DIR):
    #sys.stderr.write(filename + '\n')
    if filename.endswith(".sat"):
        if pos_num >= NUM:
            continue
        pos_num += 1
        example = np.loadtxt(EXAMPLES_DIR + '/' + filename, delimiter=' ', skiprows=0)
        #example = np.asmatrix(example)
        #example = np.resize(example,(430,3))
        X.append(example)
        #print(example.shape)
        shape = example.shape
        #X = np.append(X, example, axis=0)
        Y.append(1)
    elif filename.endswith(".unsat"):
        if neg_num >= NUM:
            continue 
        example = np.loadtxt(EXAMPLES_DIR + '/' + filename, delimiter=' ', skiprows=0)
        #example = np.resize(example,(430,3))
        #example = np.asmatrix(X, example, axis=0)        
        neg_num += 1
        X.append(example)
        #X = np.vstack(X, example)
        Y.append(0)


X = np.asarray(X)
#for i in X:
#    print(i.shape)

#sys.exit()
Y = np.asarray(Y)

#Y = keras.utils.to_categorical(Y, num_classes=2)

#load data, separating by comma; skip the first row
#print(dataset)


#use last column (column 5) as the label (one for each example/set of features)
#print("Labels:")
#print(Y)

#Split into train (90%)  and test (10%) data.
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.1)

#Create a simple feed-forward model.
model = Sequential()
#input dimension should be same as the number of features


model.add(Reshape((NUM_CLAUSES, NUM_VARIABLES + 1, 1)))
model.add(Conv2D(64, (50, 50), padding='same'))
model.add(Dropout(0.3))
model.add(Flatten())

#model.add(LSTM(32, init='uniform', activation='relu'))
#model.add(Dense(32, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model for 200 epochs
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=100, batch_size=20)

#Look at the results
scores = model.evaluate(X_test, Y_test)
sys.stdout.write("Negative Examples " + str(neg_num) + "\n")
sys.stdout.write("Positive Examples " + str(pos_num) + "\n")
print("Accuracy: %.2f%%" % (scores[1]*100))

