import json
import sys
from main import load_data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception(
            "You made a mistake! Incorrect arguments - please enter a valid file path as the single argument.")

    input_file_path = sys.argv[1]
    A, B, PI, state_list = load_data(input_file_path, is_tagged=True, calc_acc=False)

    save_dict = {
        "A_matrix": A,
        "B_matrix": B,
        "PI_matrix": PI,
        "state_list": list(state_list)
    }
    with open('hmmmodel.txt', 'w') as write_file:
        write_file.write(json.dumps(save_dict))

    print(f"Finished training on: {input_file_path}")