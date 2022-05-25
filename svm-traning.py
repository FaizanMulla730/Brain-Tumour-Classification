import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
path = os.listdir('C:/Users/HP/OneDrive/Desktop/BT/Final Code100%/brain tumor/input_data_resized/training_set/')
classes = {'Normal_Brain': 0, 'Brain_Tumor_Glioma Level':1, 'Brain_Tumor_meningioma Level':2,'Brain_Tumor_Pitutorial':3}
import cv2
X = []
Y = []
for cls in classes:
    pth = 'C:/Users/HP/OneDrive/Desktop/BT/Final Code100%/brain tumor/input_data_resized/training_set/' + cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
       # print(img)
        img = cv2.resize(img, (100,100))
        X.append(img)
        Y.append(classes[cls])
np.unique(Y)
print(np.unique(Y))
X = np.array(X)
Y = np.array(Y)
a=pd.Series(Y).value_counts()
print(a)
X.shape
print(X.shape)
with plt.xkcd():
    plt.imshow(X[0], cmap='Blues')
    X_updated = X.reshape(len(X), -1)
a=X_updated.shape
print(a)
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=0.20)
b=xtrain.shape, xtest.shape
print(b)
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


from sklearn.decomposition import PCA
print(xtrain.shape, xtest.shape)
pca = PCA(.98)
pca_train = pca.fit_transform(xtrain)
pca_test = pca.transform(xtest)
c=pca_test.shape, pca_train.shape
print(c)
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
sv = SVC()
sv.fit(pca_train, ytrain)

pca_train.shape
print("training score: ", sv.score(pca_train, ytrain)*100)
print("testing score: ", sv.score(pca_test, ytest)*100)