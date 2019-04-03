#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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


#print features_train[0]
#print features_test[0]

#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
print clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print clf.score(features_test, labels_test)

#########################################################


import numpy as np

#########################################################
## Find out feature/column that has least number of zeros to use for normal distribution.
saraFeaturesArray = [x for i, x in enumerate(features_train) if labels_train[i] == 0]
saraFeaturesNP = np.array(saraFeaturesArray)
print type(saraFeaturesNP)
print type(features_train)
fs = saraFeaturesNP.shape
print saraFeaturesNP.shape, features_train.shape
stdSara = np.std(saraFeaturesNP, axis=0)
maxSTDIndexSara = stdSara.argmax(axis=0)
print len(stdSara), np.max(stdSara), maxSTDIndexSara
#print [x for x in saraFeaturesNP[457] if x != 0]
#########################################################


#########################################################
# #Plotting normal distribution (Chandra)
# import numpy as np
# #samples = np.random.normal(size=10000)

# bins = np.linspace(-5, 5, 30)
# histogram, bins = np.histogram(maxSTDIndexSara, bins=bins, normed=True)

# bin_centers = 0.5*(bins[1:] + bins[:-1])
# #print bin_centers

# from scipy import stats
# pdf = stats.norm.pdf(bin_centers)

# import matplotlib.pyplot as plt
# #from matplotlib import pyplot as plt
# plt.figure(figsize=(6, 4))
# plt.plot(bin_centers, histogram, label="Histogram of samples")
# plt.plot(bin_centers, pdf, label="PDF")
# plt.legend()
# plt.show()

######### Or ######################

from scipy import stats
import matplotlib.pyplot as plt

# Distribution fitting
# norm.fit(data) returns a list of two parameters 
# (mean, parameters[0] and std, parameters[1]) via a MLE approach 
# to data, which should be in array form.
parameters = stats.norm.fit(saraFeaturesNP[maxSTDIndexSara])
print parameters

# now, parameters[0] and parameters[1] are the mean and 
# the standard deviation of the fitted distribution
x = np.linspace(-5,5,30)

# Generate the pdf (fitted distribution)
fitted_pdf = stats.norm.pdf(x, loc = parameters[0], scale = parameters[1])
# Generate the pdf (normal distribution non fitted)
normal_pdf = stats.norm.pdf(x)

# Type help(plot) for a ton of information on pyplot
plt.plot(x,fitted_pdf,"red",label="Fitted normal dist",linestyle="dashed", linewidth=2)
plt.plot(x,normal_pdf,"blue",label="Normal dist", linewidth=2)
plt.hist(saraFeaturesNP[maxSTDIndexSara],normed=1,color="cyan",alpha=.3) #alpha, from 0 (transparent) to 1 (opaque)
plt.title("Normal distribution fitting")
# insert a legend in the plot (using label)
plt.legend()

# we finally show our work
plt.show()
#########################################################

