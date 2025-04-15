import json
import torch
from torch.utils.data import Dataset, random_split
from models.tests import build_scale
import numpy as np

"""
Vocabulary adapted from:
    https://abcnotation.com/wiki/abc:standard:v2.2
"""
NOTES = "ABCDEFGabcdefg"
ACCIDENTALS = "^_="
DECORATIONS = "~HLMOPSTuv"
DOT = "."
RESTS = "z"
BARS = ['|', '||', '|:', ':|', ':||']
TIE = '-'
SLUR = '()'
STRUCTURAL = ['K:', 'L:', 'M:']
DURATIONS = "23468"
SPECIAL = {
    'PAD': '[PAD]',
    'UNK': '[UNK]',
    'START': '[START]',
    'END': '[END]'
}

PREFIXED = [f"{p}{note}" for note in (NOTES + RESTS) for p in ACCIDENTALS + DOT + DECORATIONS]
SUFFIXED = [f"{note}{duration}" for note in (NOTES + RESTS) for duration in DURATIONS + DOT]

TRIPLET_PREFIX = [f"3{a}{b}{c}" for a in NOTES for b in NOTES for c in NOTES]

OCTAVE_NOTES = [f"{note}," for note in (NOTES + RESTS)]
OCTAVE_NOTES += [f"{note}'" for note in (NOTES + RESTS)]

FRACTIONAL_NOTES = [f"{note}/{dur}" for note in (NOTES + ''.join(SUFFIXED)) for dur in DURATIONS]
FRACTIONAL_RESTS = [f"z/{dur}" for dur in DURATIONS]

VOCAB = list(SPECIAL.values()) + list(NOTES) + PREFIXED + SUFFIXED + STRUCTURAL + BARS +\
        [TIE, '(', ')'] + TRIPLET_PREFIX + OCTAVE_NOTES + FRACTIONAL_RESTS

VOCAB_SIZE = len(VOCAB)
# print(f"Vocabulary size: {VOCAB_SIZE}")


PAD = SPECIAL['PAD']
UNK = SPECIAL['UNK']
START = SPECIAL['START']
END = SPECIAL['END']

EXCLUDED_ENTRIES = {"1850"}
EXCLUDED_KEYS = {'id', 'title', 'book', 'transcription', 'notes', 'annotations'}



def tok2ind(token):
    try:
        return VOCAB.index(token)
    except ValueError:
        return VOCAB.index(UNK)


def ind2tok(index):
    if 0 <= index < len(VOCAB):
        return VOCAB[index]
    return UNK


def tokenize_melody(melody):
    all_tokens = []

    for line in melody:
        line = line.strip().replace('x', 'z')   # normalize 'x' to 'z'
        if line.startswith('C:'):
            composer = line[2:].strip()
            all_tokens.append(f"C:{composer} ") if composer not in all_tokens else None
            continue
        if line.startswith('R:'):
            rhythm = line[2:].strip()
            all_tokens.append(f"R:{rhythm} ") if rhythm not in all_tokens else None
            continue

        line_tokens = tokenize_line(line)
        all_tokens.extend(line_tokens)

    return all_tokens


def tokenize_line(line):
    tokens = []
    i = 0

    vocab_sort = sorted(VOCAB, key=len, reverse=True)
    while i < len(line):
        match = None

        if i + 3 <= len(line) and line[i:i+3] in TRIPLET_PREFIX:
            match = line[i:i+3]
            tokens.append(match.strip())
            i += 3
            continue

        for token in vocab_sort:
            if line.startswith(token, i):
                match = token
                break

        if match:
            tokens.append(match.strip())
            i += len(match)
        else:
            # Handle special cases
            if i + 2 < len(line) and line[i + 1] == '/' and line[i + 2].isdigit():
                # fractional duration
                note = line[i]
                fraction = line[i:i + 3]
                tokens.append(note)
                tokens.append(fraction)
                i += 3
            else:
                tokens.append(UNK)
                i += 1

    return tokens


def entry2tensor(entry):
    tokens = []
    metadata = {}

    if 'title' in entry:
        val = entry['title']
        tokens.append(f"title:{val} ")
        metadata['title:'] = val

    if 'time_signature' in entry:
        val = entry['time_signature']
        tokens.append(f"T:{val} ")
        metadata['T:'] = val

    if 'note_length' in entry:
        val = entry['note_length']
        tokens.append(f"L:{val} ")
        metadata['L:'] = val

    if 'key' in entry:
        val = entry['key']
        tokens.append(f"K:{val} ")
        metadata['K:'] = val

    tokens.append(f'{START} ')

    melody = entry.get('melody', '')
    melody_tokens = tokenize_melody(melody)
    tokens.extend(melody_tokens)

    tokens.append(END)
    # print(f"Tokens: {''.join(tokens)}")

    indices = [tok2ind(tok) for tok in tokens]

    return torch.tensor(indices, dtype=torch.long), metadata


class ABCDataset(Dataset):
    def __init__(self, json_file, augment_data=False, transpose_range=(-2, 3)):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.file_name = json_file
        self.sequences = []
        self.entry_indices = list(self.data.keys())
        self.metadata = []
        self.vocab_size = VOCAB_SIZE
        self.vocab = VOCAB
        self.pad_token = PAD
        self.pad_idx = VOCAB.index(PAD)
        self.entries = []

        self.idx2char_dict = {i: ch for i, ch in enumerate(VOCAB)}

        self.augment_data = augment_data
        self.transpose_range = transpose_range

        for idx in self.entry_indices:
            if str(idx) in EXCLUDED_ENTRIES:
                continue
            entry = self.data[idx]
            self.entries.append(entry)
            entry_t, metadata = entry2tensor(entry)
            self.sequences.append(entry_t)
            self.metadata.append(metadata)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_tensor = self.sequences[idx]
        entry = self.entries[idx]
        key = entry.get('K:', 'C')
        if self.augment_data:
            shift = np.random.randint(*self.transpose_range)
            sequence_tensor = self.transpose_tensor(sequence_tensor, shift, key)

        input_tensor = sequence_tensor[:-1]
        target_tensor = sequence_tensor[1:]
        metadata = self.metadata[idx]

        return input_tensor, target_tensor, metadata

    def transpose_tensor(self, tensor, steps, key):
        transposed = tensor.clone()
        for i in range(tensor.size(0)):
            idx = torch.argmax(tensor[i]).item()
            char = self.vocab[idx]
            if char in NOTES:
                shifted = self.transpose_note(char, steps, key)
                transposed[i] = 0
                transposed[i][tok2ind(shifted)] = 1
        return transposed

    def transpose_note(self, note, shift, key):
        scale = build_scale(key)
        is_upper = note.isupper()
        base = note.upper()

        if base not in scale:
            return note

        i = scale.index(base)
        new_i = (i + shift) % len(scale)
        transposed = scale[new_i]
        return transposed if is_upper else transposed.lower()

    def get_pad_idx(self):
        return self.pad_idx

    def get_vocab_size(self):
        return self.vocab_size

    def split_dataset(self, train_ratio=0.8, val_ratio=0.1):
        train_size = int(train_ratio * len(self))
        val_size = int(val_ratio * len(self))
        test_size = len(self) - train_size - val_size
        return random_split(self, [train_size, val_size, test_size])
