import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    cur_features = []
    cur_labels = []
    result = []

    for i in range(len(features)):
        cur_features.append(features[i])
        cur_labels.append(labels[i])
        if ((i + 1) % batch_size) == 0:
            result.append([cur_features, cur_labels])
            cur_features = []
            cur_labels = []
    
    ### If anything is left over, then adding that too. We'll not that there are lelftovers if 
    ### there is a remainder of division of number of features with batch size
    if (len(features) % batch_size) != 0:
        result.append([cur_features, cur_labels])

    return result


### Alternative solution:
#    output_batches = []
#    
#    sample_size = len(features)
#    for start_i in range(0, sample_size, batch_size):
#        end_i = start_i + batch_size
#        batch = [features[start_i:end_i], labels[start_i:end_i]]
#        output_batches.append(batch)
#        
#    return output_batches
