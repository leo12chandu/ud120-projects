#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy as np
import math
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL", 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
#highestSalary = 0
#highesBonus = 0
#Find outlier. It has max salary and bonues
#print np.amax(data, axis = 0)

#print data_dict.itervalues().next()

d = dict((k,v) for k, v in data_dict.items() if (v["bonus"] > 4999999 and v["salary"] > 1000000))

for key, value in data_dict.iteritems():
    if (value["bonus"] <> "NaN" and value["bonus"] > 4999999 and value["salary"] <> "NaN" and value["salary"] > 1000000):
        print "key=", key, "; bonus=", value["bonus"], "; salary=", value["salary"]

print len(data_dict.keys())
print len(d.keys())

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

