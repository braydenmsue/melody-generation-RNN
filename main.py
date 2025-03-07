import os


# after hitting K, read
def read_data(data_dir, set_name='train'):

    # read txt file
    file_path = os.path.join(data_dir, set_name + '.txt')
    with open(file_path, 'r') as f:
        data = f.read()

    return data     # string contents of txt file


def main(input_dir):
    data = read_data(input_dir)
    # print(type(data))
    print(data)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data')
    main(data_path)

