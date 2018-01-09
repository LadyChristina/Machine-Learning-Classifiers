# Loosely based on this tutorial: http://blogs.ekarshi.com/wp/2017/03/20/rbfradial-basis-function-neural-network-in-python-machine-learning/

from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from numpy.linalg.linalg import pinv
import time
from random import randint
import csv


'''
Dimensionality Reduction with PCA
Reduce data to nDimensions
'''
def reduceDimensions(x_train,x_test,nDimensions):
    pca = PCA(n_components=nDimensions)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test=pca.transform(x_test)
    #print('Explained variance ratio:',sum(pca.explained_variance_ratio_))
    return (x_train,x_test)

'''
Perform K-means clustering on x_train data with k clusters
'''
def kMeans(x_train,k):
    return KMeans(n_clusters=k).fit(x_train)    

'''
Returns numCenters integers in the range [0,size-1]
'''
def randomIndexes(size,numCenters):
    return [randint(0,size-1) for i in range(numCenters)]
    
'''
The RBF neural network
'''
class RBF():

    '''
    Finds the center for every neuron in the hidden layer
    param method: training method for the hidden layer, options: 'kmeans' or 'random'
    '''
    def getCenters(self, method = 'random'):
        if method == 'kmeans':
            self.centers = kMeans(self.x_train,k = self.numCenters).cluster_centers_
        else:
            self.centers = np.array([self.x_train[i] for i in randomIndexes(len(self.x_train),self.numCenters)])
        
    '''
    Computes the sigma variable that represents the spread 
    If the centers have been generated with K-means clustering, then sigma is equal to the standard deviation of ever cluster (different sigma for every neuron in the hidden layer)
    Since there is a chance of random generation of the centers, we have to use another formula
    We calculate sigma as the max distance between any two centers / sqrt (#centers) 
    This method produces a single sigma value for all neurons in the hidden layer and can be used along with any method of generating centers
    '''
    def getSigma(self):
        maxDist = 0
        for i in range(self.numCenters):
            for j in range(self.numCenters):
                if i==j:
                    continue
                dist = np.square(np.linalg.norm(self.centers[i] - self.centers[j]))
                if dist > maxDist:
                    maxDist = dist
        self.sigma = maxDist/np.sqrt(self.numCenters)
        
    '''
    Activation (radial basis) function
    Specifically, the Gaussian function
    '''
    def gaussian(self,x,c):
        sqNorm = np.square(np.linalg.norm(x - c))
        g = np.exp(-(sqNorm)/(np.square(self.sigma)))
        return g
    
    '''
    Activation (radial basis) function
    Specifically, the Multi-quadratic function
    '''
    def multiquadratic(self,x,c):
        sqNorm = np.square(np.linalg.norm(x - c))
        mq = np.sqrt(sqNorm + np.square(self.sigma))
        return mq
    
    '''
    Activation (radial basis) function
    Specifically, the Thin Plate Spline function
    '''
    def thinSpline(self,x,c):
        norm = np.linalg.norm(x - c)
        if norm == 0:
            # assuming that 0log0 = 0
            return 0
        sqNorm = np.square(norm)
        ts = sqNorm * np.log(norm)
        return ts
        
    '''
    Training the network
    '''
    def fit(self,x_train,y_train,numCenters,centerMethod,RBfunction):
        start_time = time.clock()
        self.x_train = x_train
        self.y_train = y_train
        self.numCenters = numCenters
        self.RBfunction = RBfunction
        self.getCenters(centerMethod)
        self.getSigma()
        output = np.zeros(shape=(len(x_train),self.numCenters))
        for i,x in enumerate(self.x_train):
            out=[]
            for c in self.centers:
                f = RBfunction(self,x,c)
                out.append(f)
            output[i] = np.array(out)
        self.weights = np.dot(pinv(output),self.y_train)
        trTime = round(time.clock() - start_time,4)
        print ("Training time: ",trTime, " seconds")
        return trTime
        
    '''
    Predicting the class of every element in the dataset x_test
    '''
    def predict(self,x_test):
        predictions = []
        for i,x in enumerate(x_test):
            out = []
            for c in self.centers:
                f = self.RBfunction(self,x,c)
                out.append(f)             
            output = np.dot(np.array(out),self.weights)
            prediction = output.argmax()
            predictions.append(prediction)
        return predictions
            
    '''
    Evaluates the performance of the network on train data and test data
    '''
    def evaluate(self,x_test,y_test):
        start_time = time.clock()
        accuracy = []
        predictions = self.predict(self.x_train)
        accuracy.append(round(self.accuracy(predictions,self.y_train),4))
        print("Accuracy on train data: ",accuracy[0] )
        predictions = self.predict(x_test)
        accuracy.append(round(self.accuracy(predictions,y_test),4))
        print('Accuracy on test data: ',accuracy[1])
        testTime = round(time.clock() - start_time,4)
        print ("Evaluation time: ",testTime, " seconds")
        return accuracy,testTime
        
    '''
    Compares the predictions of the network to the actual labels of the dataset 
    in order to compute the accuracy on given data
    '''
    def accuracy(self,predictions,labels):
        accuracy = 0
        for i,pred in enumerate(predictions):
            if pred == labels[i].argmax():
                accuracy +=1
        accuracy=(accuracy/len(predictions))
        return accuracy
    
'''
Loading and processing MNIST data
param oddEven: if True then we want to distinguish between odd and even numbers (2 output neurons) 
'''
def loadData(oddEven = False):
    #load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if oddEven:
        #change labels for odd vs even classification
        y_train = y_train % 2
        y_test = y_test % 2
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
    return x_train, y_train, x_test, y_test
    
'''
Performs tests by exhaustively combining the given possible values for each parameter of the RBF network
Saves the results in csv file specified by filename
'''
def tests(x_train, y_train, x_test, y_test,filename):
    count = 0
    pca = [88,None]
    numNeurons = [10, 100, 500, 1000]  
    trainSample = [5000]
    testSample = [1000]
    centerMethods = ['random', 'kmeans']
    functions = [RBF.gaussian,RBF.multiquadratic,RBF.thinSpline]
    with open(filename, 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['#Test','Dimensions','Training sample size', 'Testing sample size','Number of hidden neurons',
                        'Method for computing centers', 'Radial basis function','Training time (sec)','Testing time (sec)','Training accuracy','Testing accuracy'])
    for dim in pca:   
        x_train_, x_test_ = x_train,x_test
        if dim != None:            
            x_train_, x_test_ = reduceDimensions(x_train,x_test,dim)
        for sample1 in trainSample:
            x_train_ = x_train_[:sample1]
            y_train_ = y_train[:sample1]
            for sample2 in testSample:  
                x_test_ = x_test_[:sample2]
                y_test_= y_test[:sample2]            
                for num in numNeurons:  
                    for method in centerMethods:
                        for function in functions:
                            print(count+1,',', dim, "dimensions ,",sample1 ," training samples,",sample2," testing samples",num, "neurons in the hidden layer, training method: ",method,", activation function: ",function.__name__ )                  
                            rbf = RBF()                    
                            trTime = rbf.fit(x_train_,y_train_,numCenters = num,centerMethod=method,RBfunction = function)
                            accuracy,testTime = rbf.evaluate(x_test_,y_test_)
                            count+=1
                            print()
                            #We open the file every time instead of keeping it open, so that the results won't be lost in case of false termination
                            with open(filename, 'a', newline='') as csvfile:
                                csvWriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                                csvWriter.writerow([count,dim if dim != None else 784,sample1,sample2,num,method, function.__name__,trTime,testTime,accuracy[0],accuracy[1]])

''' Loading the data for digit recognition (10 output neurons) and testing the network '''  
x_train, y_train, x_test, y_test = loadData()
tests(x_train,y_train,x_test,y_test,'RbfTestResultsTenClasses.csv')

'''Loading the data for odd vs even classification (2 output neurons) and testing the network '''  
x_train, y_train, x_test, y_test = loadData(oddEven = True)
tests(x_train,y_train,x_test,y_test,'RbfTestResultsTwoClasses.csv')


