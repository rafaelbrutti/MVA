from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from utils import create_Submissioncsv, reconstructImage
from hog import hog
from kernels import RBF
from multiclassSVM import MultClassSVMClassifier_OvO


X_train = pd.read_csv('dataset/Xtr.csv', header = None, sep=',', usecols = range(3072))
X_test = pd.read_csv('dataset/Xte.csv', header = None, sep=',', usecols = range(3072))
Y_train = pd.read_csv('dataset/Ytr.csv')

X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)

newdata_train = reconstructImage(X_train)
newdata_val = reconstructImage(X_val)
newdata_test = reconstructImage(X_test)

# hog features of train set 
n_sample = len(newdata_train)
hog_train = np.zeros((n_sample,144))
for i in range(n_sample):
    hog_train[i,:] = hog(newdata_train[i])

# hog features of validation set     
n_sample = len(newdata_val)
hog_val = np.zeros((n_sample,144))
for i in range(n_sample):
    hog_val[i,:] = hog(newdata_val[i])

# hog features of test set     
n_sample = len(newdata_test)
hog_test = np.zeros((n_sample,144))
for i in range(n_sample):
    hog_test[i,:] = hog(newdata_test[i])

# Parameters
sigma = np.sqrt(1/(2*17))
C= 5.
kernel = RBF(sigma).kernel

# Model 
model = MultClassSVMClassifier_OvO(C=C, kernel=kernel)
model.fit(hog_train, y_train.Prediction.values)

# Accuracy score on the validation set
Y_val_pred = model.predict(hog_val)
score = (y_val.Prediction == Y_val_pred).sum()/y_val.shape[0]
print("Validation score : ", score)

# Prediction on test set
Y_pred = model.predict(hog_test)

# Create submission file
create_Submissioncsv(Y_pred)