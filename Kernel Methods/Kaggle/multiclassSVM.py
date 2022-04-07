import numpy as np
from kernels import KernelSVC

########### ONE versus ONE #############

class MultClassSVMClassifier_OvO(object):
    
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.classifiers = []
    
    def fit(self, X_train, y_train):

        self.nclasses = np.unique(y_train).size
        labels = np.unique(y_train)
        
        # creation of n(n-1)/2 classifiers
        for i in range(self.nclasses):
            for j in range(i+1, self.nclasses):
                
                svm = KernelSVC(C = self.C, kernel = self.kernel)
                
                # keep only labels i and j for binary classification
                indexes = np.logical_or(y_train == labels[i],y_train == labels[j])
                y_tr = np.where(y_train[indexes] == labels[i],1,-1)
                
                svm.fit(X_train[indexes], y_tr)
                self.classifiers.append([svm,labels[i],labels[j]])
                
    def predict(self, X_test):
        predicts = np.zeros((X_test.shape[0], self.nclasses))

        for [classifier,label1, label2] in self.classifiers:
            
            pred = classifier.predict(X_test)
            predicts[np.where(pred == 1),label1] +=1
            predicts[np.where(pred == -1),label2] +=1
            
        return np.argmax(predicts, axis = 1)
    
########### ONE versus ALL #############
    
class MultClassSVMClassifier_OvA(object):
     
    def __init__(self, C, kernel):
        self.C = C
        self.kernel = kernel
        self.classifiers = []
    
    def fit(self, X_train, y_train):

        self.nclasses = np.unique(y_train).size
        labels = np.unique(y_train)
        
        # creation of n classifiers
        for i in range(self.nclasses):
            
            svm = KernelSVC(C = self.C, kernel = self.kernel)
            y_tr = np.where(y_train == labels[i], 1, -1)
            svm.fit(X_train, y_tr)
            self.classifiers.append(svm)
            
    def predict(self, X_test):
        predicts = np.zeros((X_test.shape[0], self.nclasses))
        
        for count, classifier in enumerate(self.classifiers):

            # compute the score for each classifier
            predicts[:,count] = classifier.separating_function(X_test) + classifier.b
            
        return np.argmax(predicts, axis = 1)