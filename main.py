import os


# after hitting K, read
def read_data(data_dir, set_name='train'):
    file_path = os.path.join(data_dir, set_name + ".txt")
    with open(file_path, "r") as f:
        num_newlines = 0
        i=0
        entry = {}
        # all_lines = []
        for line in f:

        #     result = line
        #     if line[0] == 'K':
        #         result = line[:2] + ' ' + line[2:]
        #
        #     if line != '\n':
        #         if line[-2] == ':':
        #             continue
        #     all_lines.append(result)
        #
        # print(all_lines)
        # # write to  new file
        # with open(os.path.join(data_dir, set_name + "_new.txt"), "w") as f:
        #     f.writelines(all_lines)
        #     f.close()

            if line == '\n':
                num_newlines += 1
                if num_newlines == 2:
                    pass

            if line[0] == 'X':
                entry['id'] = line.split()[1]

            elif line[0] == 'T':
                entry['title'] = line[2:].strip()

            elif line[0] == 'M':
                entry['time_signature'] = line.split()[1]

            elif line[0] == 'L':
                entry['note_length'] = line.split()[1]
            elif line[0] == 'K':
                entry['key'] = line.split()[1]

            print(entry)
            i+=1
            if i==18:
                break

    return
    # return data     # string contents of txt file


def main(input_dir):
    data = read_data(input_dir)
    print(type(data))
    # print(data)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data')
    main(data_path)

