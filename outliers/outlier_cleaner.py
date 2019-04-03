#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    pre_cleaned_data = []
    import math;

    ### your code goes here
    print "predictions, ages and networths"
    #print predictions[0], ages[0], net_worths[0]

    pre_cleaned_data = [(ages[pi], net_worths[pi], (p - net_worths[pi]) ** 2) for pi, p in enumerate(predictions)]
    pre_cleaned_data.sort(key=lambda x: x[2])
    cleaned_data = pre_cleaned_data[:int(math.floor(len(pre_cleaned_data) * 0.9))]

    #for pi, p in enumerate(predictions):
        #print predictions[pi], ages[pi], net_worths[pi], predictions - net_worths[pi]
        #print (predictions - net_worths[pi]) ** 2
        #pre_cleaned_data = (ages[pi], net_worths[pi], (p - net_worths[pi]) ** 2)
    

    print len(pre_cleaned_data)
    return cleaned_data

