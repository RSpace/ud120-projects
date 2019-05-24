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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print("Data points:", len(enron_data), enron_data.keys())
print("Features:", len(enron_data[enron_data.keys()[0]]), enron_data[enron_data.keys()[0]].keys())
pois = {k : v for k,v in filter(lambda t: t[1]["poi"] == 1, enron_data.iteritems())}
print("Points of interest:", len(pois))
print("Total value of the stock belonging to James Prentice:", enron_data["PRENTICE JAMES"]["total_stock_value"])
print("Email messages from Wesley Colwell to persons of interest:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print("value of stock options exercised by Jeffrey K Skilling:", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

for name in ["SKILLING JEFFREY K", "FASTOW ANDREW S", "LAY KENNETH L"]:
  person = enron_data.get(name)
  if person:
    print("Total payments for", name, person["total_payments"])
  else:
    print("No data for", name)

with_salary = {k : v for k,v in filter(lambda t: t[1]["salary"] != "NaN", enron_data.iteritems())}
print("People with qualified salary", len(with_salary))
with_email = {k : v for k,v in filter(lambda t: t[1]["email_address"] != "NaN", enron_data.iteritems())}
print("People with known email address", len(with_email))

# A python dictionary can’t be read directly into an sklearn classification or regression algorithm; instead, it needs a numpy array or a list of lists (each element of the list (itself a list) is a data point, and the elements of the smaller list are the features of that point).

# We’ve written some helper functions (featureFormat() and targetFeatureSplit() in tools/feature_format.py) that can take a list of feature names and the data dictionary, and return a numpy array.

# In the case when a feature does not have a value for a particular person, this function will also replace the feature value with 0 (zero).
