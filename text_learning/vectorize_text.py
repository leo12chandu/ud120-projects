#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

def replace_all(text, lst):
    for l in lst:
        text = text.replace(l, '')
    return text

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0
words_to_replace = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        if True:#if temp_counter < 200:
            path = os.path.join('..', path[:-1])
            print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            stemmed_email = parseOutText(email)
            #print stemmed_email

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            #replaced_stemmed_email = [stemmed_email.replace(wtr, '') for wtr in words_to_replace]
            replaced_stemmed_email = replace_all(stemmed_email, words_to_replace)

            ### append the text to word_data
            word_data.append(replaced_stemmed_email)

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            person = 0 if name == 'sara' else 1
            from_data.append(person)


            email.close()

print "emails processed"
from_sara.close()
from_chris.close()
print "word_data[152]", word_data[152]

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )





### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
vectorizer = TfidfVectorizer(stop_words='english')
#print type(word_data)
#print word_data[0]
vectorized_word_data = vectorizer.fit_transform(word_data)
feature_names = vectorizer.get_feature_names()
#print feature_names[1]
print "Number of features", len(feature_names)
print "feature 34597 = ", feature_names[34597]
#print "word_data 34597 = ", word_data[34597]


