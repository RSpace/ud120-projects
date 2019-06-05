#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    data_point_count = len(predictions)
    all_data = []

    ### your code goes here
    for i in range(data_point_count):
        prediction = predictions[i]
        net_worth = net_worths[i]
        age = ages[i]
        error = net_worth - prediction
        all_data.append((age, net_worth, error))

    sorted_by_error = sorted(all_data, key=lambda tup: tup[2])
    percent_count = int(round(data_point_count * 0.1))

    return sorted_by_error[percent_count:]
