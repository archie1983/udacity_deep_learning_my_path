import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    ### assuming that Y contains labels of actual results in the given model
    ### and P contains predicted probabilities of each result by the model 
    
    # first lets get a list of ln(P) and ln(1 - P) for both situations:
    y1P = np.log(P)
    y0P = np.log(np.subtract(1, P))

    cross_entropy_list = []
    for i in range(len(Y)):
        #print(Y[i]," * ",y1P[i]," + ",(1-Y[i])," * ",y0P[i])
        cross_entropy_list.append(Y[i] * y1P[i] + (1-Y[i]) * y0P[i])

    return -1.0 * sum(cross_entropy_list)

### Alternative solution:
#def cross_entropy(Y, P):
#    Y = np.float_(Y)
#    P = np.float_(P)
#    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
