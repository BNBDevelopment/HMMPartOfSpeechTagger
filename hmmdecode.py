import json
import sys
from main import viterbize
import pandas as pd


def load_data(file_path):
    input_file = pd.read_csv(file_path, header=None, names=["sentence"], dtype=str, sep='8654356352134123123213516',
                             engine='python')
    filter_split_on_space = lambda sentence_series: sentence_series.str.split(" ")
    working_file = input_file.apply(filter_split_on_space, axis=1)

    return working_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "You made a mistake! Incorrect arguments - please enter a valid file path as the single argument.")

    input_file_path = sys.argv[1]

    with open('hmmmodel.txt', 'r') as read_file:
        save_dict = json.load(read_file)

    A = save_dict['A_matrix']
    B = save_dict['B_matrix']
    PI = save_dict['PI_matrix']
    state_list = set(save_dict['state_list'])

    working_file = load_data(input_file_path)

    with open('hmmoutput.txt', 'wb') as output_file:
        for line in working_file.iterrows():
            assignment = viterbize(line[1][0], A=A, B=B, PI=PI, state_list=state_list)

            print_line = ''
            for idx, word in enumerate(line[1][0]):
                print_line = print_line + word + "/" + assignment[idx] + " "
            print_line = (print_line.rstrip() + "\n").encode('utf-8')
            output_file.write(print_line)

    print(f"Finished decoding {input_file_path}, see output at: hmmoutput.txt")