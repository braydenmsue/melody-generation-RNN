import json
from torch.utils.data import Dataset


class ABCDataset(Dataset):
    def __init__(self, json_file, transform=None, target_transform=None):
        # Load data from JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.keys = list(self.data.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Get the key for this index
        key = self.keys[idx]

        # Get the dictionary for this item
        item_dict = self.data[key]

        # You can choose what to use as your label
        label = item_dict.get("key", None)  # Using musical key as label

        if self.transform:
            item_dict = self.transform(item_dict)
        if self.target_transform and label is not None:
            label = self.target_transform(label)

        return item_dict, label

    def get_by_id(self, item_id):
        if item_id in self.data:
            return self.data[item_id]
        return None

