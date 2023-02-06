import random
from collections import Counter

import pandas as pd
import time


def dict_increment(dict_obj, key, inc_amount=1):
    if key in dict_obj.keys():
        dict_obj[key] = dict_obj[key] + inc_amount
    else:
        dict_obj[key] = inc_amount

def process_tagged(sentence_series, state_count, state_trns_count, initial_state_count, word_pos_count, word_count, markov_order=1):
    sentence = []
    tags = []

    ##################################### initial_state_count #####################################
    first_word_pos = sentence_series[0][0].rsplit("/", 1)[1]
    dict_increment(dict_obj=initial_state_count, key=first_word_pos)

    for idx, word_slash_pos in enumerate(sentence_series[0]):
        data_tuple = tuple(word_slash_pos.rsplit("/", 1))
        sentence.append(data_tuple[0])
        tags.append(data_tuple[1])

        word = data_tuple[0]
        pos = data_tuple[1]

        ##################################### state_count #####################################
        dict_increment(dict_obj=state_count, key=pos)


        ##################################### state_trns_count #####################################
        if idx >= markov_order:
            previous_words_pos = [sentence_series[0][i].rsplit("/", 1)[1] for i in range(idx-1, idx-markov_order-1, -1)]
            transition_key = "_".join(previous_words_pos + [pos])

            dict_increment(dict_obj=state_trns_count, key=transition_key)


        ##################################### word_pos_count #####################################
        dict_increment(dict_obj=word_pos_count, key=word_slash_pos)


        ##################################### word_count #####################################
        dict_increment(dict_obj=word_count, key=word)




    return (sentence, tags)


def sum_dict_vals(d):
    return sum(d.values())

def create_B(state_count, word_count, word_pos_count):
    super_struct = {}
    columns_list = set(state_count.keys())
    rows_list = set(word_count.keys())

    word_pos_keys = set(word_pos_count.keys())

    for column in columns_list:
        row = {i: (word_pos_count[i + "/" + column] / state_count[column]) if (i + "/" + column) in word_pos_keys else B_SMOOTHING_VAL for i in rows_list}
        super_struct[column] = row
    return super_struct

def create_A(state_count, state_trns_count, dims=2):
    dim_list = set(state_count.keys())

    if dims == 2:
        super_struct = {k: dict.fromkeys(dim_list, A_SMOOTHING_VAL) for k in dim_list}
        for transition, tr_count in state_trns_count.items():
            temp = transition.split("_")
            super_struct[temp[0]][temp[1]] = tr_count / state_count[temp[0]]
        return super_struct

    elif dims == 3:
        pass
        # super_struct = {}
        # for column in dim_list:
        #     for row in dim_list:
        #         depth = {i: (word_pos_count[i + "/" + column] / state_count[column]) if (i + "/" + column) in word_pos_keys else A_SMOOTHING_VAL for i in dim_list}
        #     super_struct[column][row] = depth
        # return super_struct

def dict_average_each_over(collection_items_to_avg, dict_sum_or_val_divisor):

    if hasattr(dict_sum_or_val_divisor, "__iter__"):
        coll_sum = sum_dict_vals(dict_sum_or_val_divisor)
    else:
        coll_sum = dict_sum_or_val_divisor

    return {k: (v / coll_sum) for k, v in collection_items_to_avg.items()}

def viterbize(input_seq, A, B, PI, state_list):
    viterbi_matrix = {}
    prev_idx_pointer = {}

    input_words = input_seq

    top_k_storage = {}

    for state in state_list:
        if input_words[0] in B[state].keys():
            viterbi_matrix[state] = [PI[state] * B[state][input_words[0]]]
        else:
            #TODO - strategies
            #mean (my tests show much better acc):
            b_val = sum(B[state].values()) / len(B[state])
            #random:
            #b_val = B[state][random.choice(list(B[state].keys()))]

            viterbi_matrix[state] = [PI[state] * b_val]
        prev_idx_pointer[state] = [0]

    for observation_idx in range(1, len(input_words)):
        for state in state_list:

            #calc prev row vals:
            max_found = -1
            max_prev_state = ''
            for idx, s_prime in enumerate(state_list):

                if input_words[observation_idx] in B[state].keys():
                    val = viterbi_matrix[s_prime][observation_idx - 1] * A[s_prime][state] * B[state][input_words[observation_idx]]
                else:
                    # TODO - strategies
                    # mean (my tests show much better acc):
                    b_val = sum(B[state].values()) / len(B[state])
                    # random:
                    # b_val = B[state][random.choice(list(B[state].keys()))]
                    val = viterbi_matrix[s_prime][observation_idx-1] * A[s_prime][state] * b_val

                if val > max_found: #TODO: less than or equal?
                    max_found = val
                    max_prev_state = s_prime

            viterbi_matrix[state].append(max_found)
            prev_idx_pointer[state].append(max_prev_state)

    #print("Backtrack - Finding path...")
    max_final = -1
    max_final_state = ''
    for idx, s in enumerate(state_list):
        if viterbi_matrix[s][-1] > max_final:
            max_final = viterbi_matrix[s][-1]
            max_final_state = s

    path = [max_final_state]
    working_state = max_final_state
    for o in range(len(input_words)-1, 0, -1):
        working_state = prev_idx_pointer[working_state][o]
        path.append(working_state)

    path.reverse()
    return path

def create_PI(state_count, initial_state_count, line_count):
    temp_PI = dict_average_each_over(initial_state_count, line_count)

    return {k : temp_PI[k] if k in temp_PI.keys() else 0 for k,v in state_count.items()}


def load_data(file_path, is_tagged=False, calc_acc=False):
    time_0A = time.time()*1000
    input_file = pd.read_csv(file_path, header=None, names=["sentence"], dtype=str, sep='8654356352134123123213516', engine='python')

    time_0B = time.time()*1000
    filter_split_on_space = lambda sentence_series : sentence_series.str.split(" ")
    working_file = input_file.apply(filter_split_on_space, axis=1)

    #TODO: remove special chars?

    time_00 = time.time()*1000
    if is_tagged:

        state_count = {}                  #the count for each state type (ex: verb)
        state_trns_count = {}             #state_transition_count for calculating A
        initial_state_count = {}          #count of each state as the first word
        word_pos_count = {}               #count of each word for each pos (ex: back_verb & back_noun are different counts)
        word_count = {}                   #count of each word, ignoring pos (ex: back_verb & back_noun are part of same count)
        line_count = len(working_file)    #A count of the total lines

        working_file = working_file.apply(process_tagged, axis=1,
                                          state_count=state_count,
                                          state_trns_count=state_trns_count,
                                          initial_state_count=initial_state_count,
                                          word_pos_count=word_pos_count,
                                          word_count=word_count,
                                          markov_order=1)

        time_01 = time.time()*1000
        PI = create_PI(state_count=state_count, initial_state_count=initial_state_count, line_count=line_count)
        time_02 = time.time()*1000
        B = create_B(state_count=state_count, word_count=word_count, word_pos_count=word_pos_count)
        time_03 = time.time()*1000
        A = create_A(state_count=state_count, state_trns_count=state_trns_count)
        time_04 = time.time()*1000

        state_list = set(state_count.keys())
    if calc_acc:
        correct = 0
        total = 0
        for line in working_file:
            assignment = viterbize(line[0], A=A, B=B, PI=PI, state_list=state_list)
            labels = line[1]

            for i in range(len(labels)):
                if assignment[i] == labels[i]:
                    correct += 1
                total += 1
        print(f"Accuracy: {round(correct/total, 7) * 100}% with {correct} correct out of {total} total.")
    time_05 = time.time()*1000


    # print(f"Read CSV took: {-(time_0A - time_0B)} ms")
    # print(f"Space split took: {-(time_0B - time_00)} ms")
    # print(f"Create counts took: {-(time_00 - time_01)} ms")
    # print(f"Create PI took: {-(time_01 - time_02)} ms")
    # print(f"Create B took: {-(time_02 - time_03)} ms")
    # print(f"Create A took: {-(time_03 - time_04)} ms")
    # print(f"Viterbize each line took: {-(time_04 - time_05)} ms")
    print(f"TOTAL TIME: {-(time_0A - time_05) / 1000} seconds")

    #print("#"*50 + "Finished!" + "#"*50)
    return A, B, PI, state_list


B_SMOOTHING_VAL =  0.0000000001000001
A_SMOOTHING_VAL =  0.0000000001000001
PI_SMOOTHING_VAL = 0.0000000001000001
SCALING = 1
TOP_K_NUM = 3

# load_data("it_isdt_dev_tagged.txt", is_tagged=True, calc_acc=True)
# load_data("it_isdt_train_tagged.txt", is_tagged=True, calc_acc=True)
# load_data("ja_gsd_dev_tagged.txt", is_tagged=True, calc_acc=True)
# load_data("ja_gsd_train_tagged.txt", is_tagged=True, calc_acc=True)