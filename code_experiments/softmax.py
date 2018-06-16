import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    import math as m
    denominator = 0.0
    result_list = []

    for i in range(len(L)):
        denominator += m.exp(L[i])

    for i in range(len(L)):
        result_list.append(m.exp(L[i]) / denominator)

    return result_list
