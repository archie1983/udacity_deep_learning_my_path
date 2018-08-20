import numpy as np

# AE: We need to create a batching function according to the task rules as in the TV script course project.
# AE: Rules are outlined in rules.txt. This is where I try to achieve that.
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    number_of_batches = len(int_text) // (batch_size * seq_length)
    # AE: truncating the text to what we will be using
    usable_text = int_text[:(number_of_batches * batch_size * seq_length)]

    batch_number = 0
    
    # AE: First let's turn the usable text into some shape that resembles the required
    # AE: split into batches. We're skipping the second parameter in the reshape function
    # AE: of array because we're now only dealing with the inputs. We'll deal with
    # AE: the outputs later.
    split_inputs = np.array(usable_text).reshape(number_of_batches, batch_size, seq_length)

    # AE: We now have an almost correct split for the inputs, but not quite what was
    # AE: required in the task. For example this is what we get on the example input
    # AE: as given in the task rules:
    # usable_text.reshape(3, 1, 3, 2)
    # array([[[[ 1,  2],
    #          [ 3,  4],
    #          [ 5,  6]]],
    #
    #        [[[ 7,  8],
    #          [ 9, 10],
    #          [11, 12]]],
    #
    #        [[[13, 14],
    #          [15, 16],
    #          [17, 18]]]])
    #
    # AE: What may be obvious now, is that the the result that we want is almost a transpose of
    # AE: the tensor that we have here. What we want is this:
    # [[[  1.   2.]
    #   [  7.   8.]
    #   [ 13.  14.]]
    #
    #  [[  3.   4.]
    #   [  9.  10.]
    #   [ 15.  16.]]
    #
    #  [[  5.   6.]
    #   [ 11.  12.]
    #   [ 17.  18.]]]
    # 
    # AE: So if only we could treat the most inner rows as single units (e.g. tuples (1, 2), (3, 4), etc)
    # AE: instead of each value being separate, then the transpose would actually work. Sadly this is a 3D
    # AE: tensor and doing a transpose like <split_inputs.T> will transpose all 3 dimensions and yield this:
    #
    # [[[  1.   7.  13.]
    #   [  3.   9.  15.]
    #   [  5.  11.  17.]]
    # 
    #  [[  2.   8.  14.]
    #   [  4.  10.  16.]
    #   [  6.  12.  18.]]]
    # 
    # AE: But we really want to leave the 3rd dimension alone and treat it as a unitary element. So we only 
    # AE: want to transpose the first two dimensions. After some research it turns out that it is possible.
    # AE: by using text_splits_rs.transpose(1, 0, 2)
    inputs = split_inputs.transpose(1, 0, 2).reshape(number_of_batches, batch_size, seq_length)
    print("AE: ", inputs, " :AE")

    # AE: Now we want to prepare targets (the same as <usable_text> only shifted to the right by 1)
    print(usable_text)
    usable_targets = usable_text[1:] + [usable_text[0]]
    print(usable_targets)
    # AE: And now we organise this data into the correct shape just as before with the inputs:
    split_targets = np.array(usable_targets).reshape(number_of_batches, batch_size, seq_length)
    targets = split_targets.transpose(1, 0, 2).reshape(number_of_batches, batch_size, seq_length)
    print("AE: ", targets, " :AE")

    # AE: Now we need to combine the inputs and targets into a unified data structure

    # AE: Our final data structure, where we'll be storing the input data and output data as
    # AE: per spec in the task rules:
    final_data = np.ndarray((number_of_batches, 2, batch_size, seq_length))

    # AE: Now let's iterate through batches and batch sizes and put the data where it belongs
    for nb in range(number_of_batches):
        for bs in range(batch_size):
            final_data[nb, 0, bs] = inputs[nb, bs]
            final_data[nb, 1, bs] = targets[nb, bs]

    print("AE @ ", final_data, " @ AE")
    # text_splits = np.zeros((number_of_batches, batch_size * seq_length))
    # text_splits_sequenced = np.zeros((number_of_batches, batch_size))
    # split_n = 0
    # for ndx in range(0, len(usable_text), batch_size * seq_length):
    #     #text_splits = np.array(usable_text[ndx:ndx + batch_size * seq_length])
    #     #print(split_n, " : ", text_splits)
    #     text_splits[split_n] = np.array(usable_text[ndx:ndx + batch_size * seq_length])
        
    #     split_n += 1
    # print(text_splits)
    # text_splits_rs = text_splits.reshape(number_of_batches, batch_size, seq_length)
    # print(text_splits_rs)
    # print(text_splits_rs.transpose(1, 0, 2))
    # print(text_splits_rs.transpose(1, 0, 2).reshape(number_of_batches, batch_size, seq_length))
    # print(text_splits_rs.T)

    ### AE: Below are my previous rather fruitless attempts to achieve the same
    ### AE: by iterating through the data and somehow manually sorting it into the
    ### AE: right structure
    ## AE: We'll iterate through the whole text in batches. ndx will now be
    ## AE: iterating over a list with the start indexes for each batch
    #for ndx in range(0, len(usable_text), batch_size * seq_length):
    #    # going through each batch
    #    
    #    # inputs first
    #    this_batch_raw = np.array(usable_text[ndx:ndx + batch_size * seq_length])
    #    split_inputs[batch_number, 0] = this_batch_raw.reshape(batch_size, seq_length)
    #    
    #    # now targets
    #    this_batch_raw = np.array(usable_text[ndx + 1:ndx + 1 + batch_size * seq_length])
    #    split_inputs[batch_number, 1] = this_batch_raw.reshape(batch_size, seq_length)
    #    # just what to do now if we don't have enough members in the array for the targets
    #    
    #    # next batch
    #    batch_number += 1
    
    

    ## AE: we will iterate through the usable text in chunks of batch_size * seq_length
    ## AE: batch_starts will be a list of indexes at which we will take <batch_size * seq_length>
    ## AE: number of words. The list that we'll take from each of these indexes to the next,
    ## AE: will contain one element for each batch. So we will need to split it to all the batches.
    ## AE: Ww'll do that with the help of <segment_count>
    #batch_starts = range(0, len(usable_text), batch_size * seq_length)
    #segment_count = 0
    #
    #for bs in range(batch_starts):
    #    # AE: <batch_slices_raw> will contain one element for each batch. The elements will be gotten
    #    # AE: from the <usable_text> list starting at the current <bs> index and of length that will
    #    # AE: yield enough data for all batches.
    #    batch_slices_raw = np.array(usable_text[bs:bs + batch_size * seq_length]).reshape(batch_size, seq_length)
    #    
    #    for batch_number in range(number_of_batches):
    #        split_inputs[batch_number, 0, segment_count] = batch_slices_raw[batch_number]
    #
    #    segment_count += 1
    #
    
    return final_data

get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)
get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 4, 2)
