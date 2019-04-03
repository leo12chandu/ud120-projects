#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#Reduce the training data to 1%
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

#Train Data

from sklearn import svm
#linear kernel
#clf = svm.SVC(kernel="linear")
#RBF kernel
clf = svm.SVC(kernel="rbf", C=10000)

# C
# If C value is high, it tries to classfy all the training points. So it tries to be correct all the time and the hyperplane can be squiggly.
# Where as C value is low, it makes a smooth decision surface.

# Gamma Value
# If Gamma is high, on the points/dots that are close to the SVM line/hyperplane have more impact and hence the line can be squigly/curvy
# If Gamma is low, all the points that are far and close to hyperplane/line have the same influence and hence there is a higher chance of the plane to be a straight line.
# https://www.youtube.com/watch?v=m2a2K4lprQw

# We should avoid OVERFITTING.

print "SVM Fitting/training with training data"
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

#Predict from trained data.

print "predicting features_test"
t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "Calculating Accuracy"

#Calculate the accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print acc;


#Check predictions
print "10th prediction: ", pred[10];
print "26th prediction: ", pred[26];
print "50th prediction: ", pred[50];

#Chris' emails
print pred;
print "Total test emails:", len(labels_test)
predChris = pred[pred == 1] #np.where(pred == 1) #(pred == 1)
print "Chris' emails", len(predChris)
predSara = pred[pred == 0]
print "Sara' emails", len(predSara)


#########################################################


