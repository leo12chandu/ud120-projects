from __future__ import division
import math
import matplotlib.pyplot as plt


def frange(start, stop, step):
    return list(xfrange(start, stop, step))


def xfrange(start, stop, step):
   while start < stop:
       yield start
       start += step

#Entropy for 2 classes.
def get_entropy(pi):
    return (-(pi) * math.log((pi), 2)) + (-(1 - pi) * math.log((1 - pi), 2))


entropy_list = []
for i in frange(0.01, 1, 0.01):
    entropy_item = get_entropy(i)
    #entropy_list.Append(entropy_item)
    print i, entropy_item

vals_to_plot = [(i, get_entropy(i)) for i in frange(0.01, 1, 0.01)]
plt.plot(vals_to_plot)

plt.show()