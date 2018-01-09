from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import time

#Dimensionality Reduction with PCA
def reduceDimensions(x_train,x_test,nDimensions):
    print ("Reducing to",nDimensions, "dimensions" )
    pca = PCA(n_components=nDimensions)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test=pca.transform(x_test)
    print('Explained variance ratio:',sum(pca.explained_variance_ratio_))
    print()
    return (x_train,x_test)

#Performs grid search with the given parameters
def gridSearch(param_grid, cv,x_train,y_train):
	clf = GridSearchCV(SVC(), param_grid, cv=cv)
	print("Searching parameters for SVM with", param_grid[0]['kernel'][0], "kernel..")
	start_time = time.clock()
	clf.fit(x_train, y_train)
	print ("Elapsed time: ",time.clock() - start_time, " seconds")
	print()
	print("Best parameters set found on development set:")
	print()
	params = clf.best_params_
	print (params)
	print()
	print("Grid scores on development set:")
	print()
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
		      % (mean, std * 2, params))
	print()
	return params


print()
#load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
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

#reducing dimensions from 784 to 100, keeping about 91.4% of the original information
x_train100, x_test100 = reduceDimensions(x_train, x_test,100)

#we'll use only the first 3000 samples for the grid search
x_train_ = x_train100[:3000]
y_train_ = y_train[:3000]

c_range = [ 1, 10, 100, 1000 ]
gamma_range = [0.001,0.0001]
param_grid_linear = [ {'C': c_range , 'kernel' : [ 'linear' ]}]
param_grid_rbf = [ { 'C': c_range, 'gamma':gamma_range , 'kernel': ['rbf'] } ]

#performing grid search for svm with linear kernel
params_linear = gridSearch(param_grid_linear,5,x_train_,y_train_)
#performing grid search for svm with rbf kernel
params_rbf = gridSearch(param_grid_rbf,5,x_train_,y_train_)

print("SVM with linear kernel")
clf_linear = SVC(kernel="linear", C=params_linear['C'])
print("Start fitting. This may take a while..")
start_time = time.clock()
#train svm on the full training dataset
clf_linear.fit(x_train100, y_train)
print ("Elapsed time: ",time.clock() - start_time, " seconds")
print()
expected = y_test
#predict the labels of the test data
predicted = clf_linear.predict(x_test100)
#print classification report
print("Classification report for classifier %s:\n%s\n"
      % (clf_linear, metrics.classification_report(expected, predicted)))
#print confusion matrix
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print()

print("SVM with RBF kernel")
clf_rbf = SVC(kernel="rbf", C=params_rbf['C'], gamma=params_rbf['gamma'])
print("Start fitting. This may take a while..")
start_time = time.clock()
#train svm on the full training dataset
clf_rbf.fit(x_train100, y_train)
print ("Elapsed time: ",time.clock() - start_time, " seconds")
print()
expected = y_test
#predict the labels of the test data
predicted = clf_rbf.predict(x_test100)
#print classification report
print("Classification report for classifier %s:\n%s\n"
      % (clf_rbf, metrics.classification_report(expected, predicted)))
#print confusion matrix
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))




