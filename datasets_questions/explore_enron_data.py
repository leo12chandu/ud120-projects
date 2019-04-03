#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

from __future__ import division
import pickle
import math
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

#print type(enron_data)
#print type(enron_data["PRENTICE JAMES"])
#print type(enron_data["PRENTICE JAMES"]["total_stock_value"])
#print enron_data["PRENTICE JAMES"].keys()

#print len(enron_data.itervalues().next()) #this gives keys (dict inside of dict)
#print len(enron_data)

#print len(enron_data["PRENTICE JAMES"].keys())

#key, value = enron_data.iteritems().next()
#print "Key=", key, "Value=", value

#print type(value)


##for key, value in enron_data.iteritems() :
##    print key, value

#poiList = {k:v for (k,v) in enron_data.iteritems() if v["poi"] == True}
#print "Number of POIs: ", len(poiList)

#jamesList = {k:v for (k,v) in enron_data.iteritems() if "james" in k.lower()}
#print jamesList

#print "James Prentice total stock value: ", enron_data["PRENTICE JAMES"]["total_stock_value"]

#print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

#skillingList = {k:v for (k,v) in enron_data.iteritems() if "skilling" in k.lower()}
#print skillingList

#print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]


#kennethList = {k:v for (k,v) in enron_data.iteritems() if "kenneth" in k.lower()}
#print kennethList

#fastowList = {k:v for (k,v) in enron_data.iteritems() if "fastow" in k.lower()}
#print fastowList

#"James Prentice" stock
#print enron_data["PRENTICE JAMES"]["total_stock_value"]

#Wesley Colwell Email messages from
#print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

#Jeffrey Skilling stock options
#print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

#most money
#print "Jeff Skilling: ", enron_data["SKILLING JEFFREY K"]["total_payments"]
#print "Kenneth Lay: ", enron_data["LAY KENNETH L"]["total_payments"]
#print "Andrew Fastow: ", enron_data["FASTOW ANDREW S"]["total_payments"]

#type of a variable
#print type(enron_data["SKILLING JEFFREY K"]["salary"])

#for key, value in enron_data.iteritems() :
#    print key, value["salary"], value["email_address"]

#print enron_data["GRAMM WENDY L"]["salary"] == 'NaN'

#Quantified salary and email address
#salaryList = {k:v for (k,v) in enron_data.iteritems() if (v["salary"]) != 'NaN'}
#print "Quantified Salary Count:- ", len(salaryList)
#emailList = {k:v for (k,v) in enron_data.iteritems() if (v["email_address"]) != 'NaN'}
#print "Quantified email Count:- ", len(emailList)

#NaN Total payments for people
#nanTotalPayments = {k:v for (k,v) in enron_data.iteritems() if (v["total_payments"]) == 'NaN'}
#print "Percentage NaN total_payments:- ", len(nanTotalPayments)/len(enron_data)

#NaN Total payments for poi
nanPOITotalPayments = {k:v for (k,v) in enron_data.iteritems() if (v["poi"] ==  True and v["total_payments"]) == 'NaN'}
POIs = {k:v for (k,v) in enron_data.iteritems() if (v["poi"] ==  True)}
print "nanPOITotalPayments= ", len(nanPOITotalPayments), "; POIs", len(POIs), "; Percentage NaN POI total_payments:- ", len(nanPOITotalPayments)/len(POIs)

#NaN Total payments for people + 10
nanTotalPayments = {k:v for (k,v) in enron_data.iteritems() if (v["total_payments"]) == 'NaN'}
print "nanTotalPayments + 10 = ", len(nanTotalPayments) + 10, "Total People + 10 = ", len(enron_data) + 10, "Percentage NaN total_payments:- ", (len(nanTotalPayments) + 10)/(len(enron_data) + 10)

#NaN Total payments for poi
nanPOITotalPayments = {k:v for (k,v) in enron_data.iteritems() if (v["poi"] ==  True and v["total_payments"]) == 'NaN'}
POIs = {k:v for (k,v) in enron_data.iteritems() if (v["poi"] ==  True)}
print "nanPOITotalPayments + 10 = ", len(nanPOITotalPayments) + 10, "; POIs + 10 = ", len(POIs) + 10, "; Percentage NaN POI total_payments + 10:- ", (len(nanPOITotalPayments) + 10)/(len(POIs) + 10)


#print "long term incentive", enron_data["SKILLING JEFFREY K"]["long_term_incentive"]



#data = featureFormat(enron_data, )
