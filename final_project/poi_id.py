#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

## OUTLIERS
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
df = pd.DataFrame.from_dict(data_dict)

#Transpose pandas
dfSB = df.T[["salary","bonus"]].astype(float)#.dropna(how="any")
#print dfSB
#print type(dfSB.values)
#print dfSB.values  #.T.tolist()
#plt.figure()
dfSB.plot.scatter(x='salary',y='bonus',c='DarkBlue')
#plt.scatter(dfSB.salary, dfSB.bonus)

dfSB["salary"].max()
dfSB["salary"].idxmax()

## REMOVE TOTAL row/data-point
data_dict.pop("TOTAL", 0)
print "dict_len:-", len(data_dict)
my_dataset = data_dict

all_features = data_dict.itervalues().next().keys()
features_list = ['poi'] + [f for f in all_features if f != 'email_address']
print "features_list: - ", features_list

## plot without TOTAL
df = pd.DataFrame.from_dict(data_dict)
#Transpose pandas
dfSB = df.T[["salary","bonus"]].astype(float)#.dropna(how="any")
dfSB.plot.scatter(x='salary',y='bonus',c='DarkBlue')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Scale features.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)
scaled_features_train = scaler.transform(features_train)
scaled_features_test = scaler.transform(features_test)

#print len(scaled_features_train), len(features)
#print scaled_features_train[0]

# PCA identify number of features (reduce dimensionality)
#%matplotlib inline
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(scaled_features_train)
#pca.transform(scaled_features_train)
#pca?
print pca.explained_variance_ratio_
#print pca.components_

import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# PCA application with n_components=8 from above.
from sklearn.decomposition import PCA
#PCA?
pca = PCA(n_components=8)
pca.fit(scaled_features_train)
features_pca_train = pca.transform(scaled_features_train)
features_pca_test = pca.transform(scaled_features_test)

# gridsearch
print "Executing gridsearch using RandomForest.."
param_grid = {
              'criterion': ['gini', 'entropy'], 
              'max_depth': [3, None],
              'min_samples_split': [2, 3, 10],
              'min_samples_leaf': [1, 3, 10],
              'bootstrap': [True, False],
              'max_features': [1, 3, 5, 8]
             }
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
#RandomForestClassifier?


gridsearch = GridSearchCV(RandomForestClassifier(n_estimators=20), param_grid = param_grid)
gridsearch.fit(features_pca_train, labels_train)
print "Done. gridsearch estimator is"
print gridsearch.best_estimator_

# Actual classification
from sklearn.ensemble import RandomForestClassifier

clf = gridsearch.best_estimator_
#clf with 0.8 precision and 1.0 recall
# clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=3, max_features=8, max_leaf_nodes=None,
#             min_samples_leaf=10, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)
#clf with 0.8 precision and 0.8 recall
#clf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#            max_depth=3, max_features=5, max_leaf_nodes=None,
#            min_samples_leaf=10, min_samples_split=3,
#            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)

clf.fit(features_pca_train, labels_train)
pred = clf.predict(features_pca_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
score = accuracy_score(labels_test, pred)
precision = precision_score(labels_test, pred)
recall = recall_score(labels_test, pred)
print "score=", score, " precision=", precision, " recall=", recall

#print labels_test, pred

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)