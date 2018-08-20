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
    
    # AE: First let's split the usable text into some shape that resembles the required
    # AE: split into batches. We're using <1> instead of <2> as the second parameter
    # AE: of split shape because we're now only dealing with the inputs. We'll deal with
    # AE: the outputs later.
    split_inputs = np.array(usable_text).reshape(number_of_batches, 1, batch_size, seq_length)

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
    
    # AE: Our final data structure, where we'll be storing the input data and output data as
    # AE: per spec in the task rules.
    final_data = np.ndarray((number_of_batches, 2, batch_size, seq_length))

    # Now let's iterate through batches and batch sizes and put the data where it belongs
    for nb in range(number_of_batches):
        for bs in range(batch_size):
            None
            #final_data[nb, 0, bs] = split_inputs[bs % number_of_batches, 0, (nb % batch_size) + (bs // number_of_batches)]

    


    text_splits = np.zeros((number_of_batches, batch_size * seq_length))
    text_splits_sequenced = np.zeros((number_of_batches, batch_size))
    split_n = 0
    for ndx in range(0, len(usable_text), batch_size * seq_length):
        #text_splits = np.array(usable_text[ndx:ndx + batch_size * seq_length])
        #print(split_n, " : ", text_splits)
        text_splits[split_n] = np.array(usable_text[ndx:ndx + batch_size * seq_length])
        
        split_n += 1
    print(text_splits)
    text_splits_rs = text_splits.reshape(number_of_batches, batch_size, seq_length)
    print(text_splits_rs)
    print(text_splits_rs.transpose(1, 0, 2))
    print(text_splits_rs.transpose(1, 0, 2).reshape(number_of_batches, batch_size, seq_length))
    #print(text_splits.T)
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
    
    return None

get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)
get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 4, 2)
