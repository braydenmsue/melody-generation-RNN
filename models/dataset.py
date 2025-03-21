import json
import torch
from torch.utils.data import Dataset, random_split

# VOCAB_DICT = {
#     'NOTES': "ABCDEFGabcdefg",
#     'MODIFIERS': "Zz|[]/:!^_~=,.0123456789(){}<>#'\"%-+ ",
#     'LABELS': "TLKM",    # Time signature, note length, key, melody
#     'UNKNOWN': "_",
#     'PAD': "$"}

NOTES = "ABCDEFGabcdefg"
MODIFIERS = "Zz|[]/:!^_~=,.0123456789(){}<>#'\"%-+ "
LABELS = "TLKM"     # Time signature, note length, key, melody
UNKNOWN = "_"
PAD = "$"

VOCAB = PAD + NOTES + MODIFIERS + LABELS + UNKNOWN
VOCAB_SIZE = len(VOCAB)

EXCLUDED_ENTRIES = {"1850"}
EXCLUDED_KEYS = {'id', 'title', 'book', 'transcription', 'notes', 'annotations'}
KEY_DICT = {'time_signature': 'T',
            'note_length': 'L',
            'key': 'K',
            'melody': 'M'}


def char2ind(c):
    if c not in VOCAB:
        return VOCAB.find("_")
    else:
        return VOCAB.find(c)


# TODO: embeddings instead of one-hot encoding
# TODO: bar-level tokenization for melody; phrase-level for fractions and stuff
def line2tensor(line):
    tensor = torch.zeros(len(line), 1, VOCAB_SIZE)
    for idx, letter in enumerate(line):
        tensor[idx][0][char2ind(letter)] = 1
    return tensor


def entry_to_tensor(entry):
    line = ""
    for key, val in entry.items():
        if key in EXCLUDED_KEYS:
            continue
        if key == 'melody':
            val = " ".join(val)
        string = f"{KEY_DICT[key]}{val}"
        line += string + " "

    # print('\n\n\n')
    # print(f'Entry: {line}')

    return line2tensor(line)


class ABCDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.file_name = json_file
        self.sequences = []
        self.entry_indices = list(self.data.keys())
        self.vocab_size = VOCAB_SIZE
        self.vocab = VOCAB
        self.pad_token = '$'
        self.pad_idx = self.vocab.find(self.pad_token)

        self.idx2char_dict = {i: ch for i, ch in enumerate(VOCAB)}

        for idx in self.entry_indices:
            if str(idx) in EXCLUDED_ENTRIES:
                continue
            entry = self.data[idx]
            entry_t = entry_to_tensor(entry)
            self.sequences.append(entry_t)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_tensor = self.sequences[idx]

        input_tensor = sequence_tensor[:-1]
        target_tensor = sequence_tensor[1:]

        return input_tensor, target_tensor

    def get_pad_idx(self):
        return self.pad_idx

    def get_vocab_size(self):
        return self.vocab_size

    def split_dataset(self, test_ratio=0.8):
        test_size = int(test_ratio * len(self))
        train_size = len(self) - test_size
        return random_split(self, [train_size, test_size])


