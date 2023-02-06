import sys

input_file_path = sys.argv[1]
with open(input_file_path, "r", encoding="utf-8") as in_f_01:
    with open("hmmoutput.txt", "r", encoding="utf-8") as in_f_02:
        total = 0
        correct = 0

        for line_tuple in zip(in_f_01.readlines(), in_f_02.readlines()):
            mine = line_tuple[1].split(" ")
            theirs = line_tuple[0].split(" ")

            for my_word, their_word in zip(mine, theirs):
                if my_word == their_word:
                    correct += 1
                total += 1

        print(f"Accuracy: {round(correct / total, 7) * 100}% with {correct} correct out of {total} total.")