"""
@author: Christina Ovezik
code adapted from Keras Documentation examples
using Keras and Tensorflow
"""

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.utils import to_categorical
from sklearn.decomposition import PCA
import time

#Dimensionality Reduction with PCA
def reduceDimensions(x_train,x_test,nDimensions):
    pca = PCA(n_components=nDimensions)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test=pca.transform(x_test)
    return (x_train,x_test)

#Creation, training and testing of the network
def runNetwork(units,input_shape,hidden_layers,nClasses,x_train,y_train,x_test,y_test,epochs=20,activation='relu',dropout_rate=0,batch_size=128,learning_rate=0.001):
    start_time = time.clock()
    #create the network
    model = Sequential()
    #add the first hidden layer
    model.add(Dense(units, activation=activation, input_shape=input_shape))
    if dropout_rate>0:
        model.add(Dropout(dropout_rate))
    #add the remaining hidden layers
    for i in range(hidden_layers-1):
        #we presume that we want every hidden layer to have the same number of neurons
        model.add(Dense(units, activation=activation))   
        if dropout_rate>0:
            model.add(Dropout(dropout_rate))
    #add visible layer
    model.add(Dense(nClasses, activation='softmax'))
    #configure the network
    model.compile(optimizer=optimizers.RMSprop(lr= learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    #train the network
    history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs , verbose=2, 
                   validation_data=(x_test, y_test))
    #evaluate the trained network on train and test data
    [train_loss,train_acc]=model.evaluate(x_train,y_train,verbose=0)
    [test_loss, test_acc] = model.evaluate(x_test, y_test,verbose=0)
    print()
    print("Evaluation result on Train Data : Loss = {}, Accuracy = {}".format(train_loss, train_acc))
    print("Evaluation result on Test Data : Loss = {}, Accuracy = {}".format(test_loss, test_acc))
    print ("Execution time: ",time.clock() - start_time, " seconds")
    #plot(history)
    
#Plotting the learning curves
def plot(history):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Testing Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Testing Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    
#load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
nClasses = 10 #number of unique labels
#reshape the data from matrix to vector
x_train = x_train.reshape((60000, 28 * 28))
x_test = x_test.reshape((10000, 28 * 28))
# change to float datatype
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# min-max normalization [0,255] -> [0,1]
x_train /= 255
x_test /= 255
# convert the labels from integers to vectors 5->[0000010000]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#an example with our final parameters
runNetwork(128,(784,),2,nClasses,x_train,y_train,x_test,y_test,dropout_rate=0.2)
