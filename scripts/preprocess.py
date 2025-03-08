import os
import json
import re
from collections import defaultdict

# Define paths
RAW_DATA_PATH = "data/raw/abc_notations.txt"  # Path to the raw ABC notation file
PROCESSED_DATA_PATH = "data/processed/standardized_abc.txt"  # Path to save the standardized data
LOOKUP_TABLE_PATH = "data/lookup_tables/songs_dict.json"  # Path to save the parsed songs dictionary

# Create directories if they don't exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOOKUP_TABLE_PATH), exist_ok=True)

# Tokens for annotations and bars
START_ANN_TOKEN = "<START_ANN>"
END_ANN_TOKEN = "<END_ANN>"
BAR_SEPARATOR = "|"

def load_raw_data(file_path):
    """
    Loads the raw ABC notation data from the specified file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Oops! The file {file_path} doesn't exist. Double-check the path.")
    
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    return lines

def standardize_formatting(lines):
    """
    Cleans up the raw ABC notation lines and adds tokens to annotations.
    Also removes redundant lines and adds bar separators.
    """
    standardized_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Handle annotations (lines starting with '%')
        if line.startswith("%"):
            line = f"{START_ANN_TOKEN} {line} {END_ANN_TOKEN}"
        
        # Handle metadata lines (e.g., X:, T:, M:, etc.)
        if any(line.startswith(prefix) for prefix in ["X:", "T:", "M:", "L:", "K:", "B:", "N:", "Z:"]):
            standardized_lines.append(line)
            continue
        
        # Add bar separators and clean up melody lines
        if "|" in line:
            bars = line.split("|")
            cleaned_bars = [re.sub(r"[{}()<>^=_]", "", bar).strip() for bar in bars]  # Remove special symbols
            line = BAR_SEPARATOR.join(cleaned_bars)
        
        standardized_lines.append(line)
    
    return standardized_lines

def parse_abc_data(lines):
    """
    Parses the standardized ABC notation into a structured dictionary.
    Each song is represented as a dictionary with fields:
    - 'id', 'title', 'time_signature', 'note_length', 'key', 'book', 'notes', 'transcription', 'annotations', 'melody'.
    """
    songs_dict = defaultdict(dict)
    current_song_id = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Extract song ID (lines starting with 'X:')
        if line.startswith("X:"):
            current_song_id = line.split(":")[1].strip()
            songs_dict[current_song_id]["id"] = current_song_id
        
        # Extract title (lines starting with 'T:')
        elif line.startswith("T:"):
            songs_dict[current_song_id]["title"] = line.split(":")[1].strip()
        
        # Extract time signature (lines starting with 'M:')
        elif line.startswith("M:"):
            songs_dict[current_song_id]["time_signature"] = line.split(":")[1].strip()
        
        # Extract note length (lines starting with 'L:')
        elif line.startswith("L:"):
            songs_dict[current_song_id]["note_length"] = line.split(":")[1].strip()
        
        # Extract key (lines starting with 'K:')
        elif line.startswith("K:"):
            songs_dict[current_song_id]["key"] = line.split(":")[1].strip()
        
        # Extract book (lines starting with 'B:')
        elif line.startswith("B:"):
            songs_dict[current_song_id]["book"] = line.split(":")[1].strip()
        
        # Extract notes (lines starting with 'N:')
        elif line.startswith("N:"):
            note_content = line.split(":")[1].strip()
            if note_content:  # Only overwrite if the line is not empty
                songs_dict[current_song_id]["notes"] = note_content  # Overwrite instead of appending
        
        # Extract transcription info (lines starting with 'Z:')
        elif line.startswith("Z:"):
            transcription_content = line.split(":")[1].strip()
            if transcription_content:  # Only overwrite if the line is not empty
                songs_dict[current_song_id]["transcription"] = transcription_content  # Overwrite instead of appending
        
        # Extract annotations (lines with <START_ANN> and <END_ANN> tokens)
        elif START_ANN_TOKEN in line:
            annotation = re.sub(f"{START_ANN_TOKEN}|{END_ANN_TOKEN}", "", line).strip()
            if annotation:  # Only append if the line is not empty
                if "annotations" not in songs_dict[current_song_id]:
                    songs_dict[current_song_id]["annotations"] = []
                songs_dict[current_song_id]["annotations"].append(annotation)
        
        # Extract melody (all other lines)
        else:
            if "melody" not in songs_dict[current_song_id]:
                songs_dict[current_song_id]["melody"] = []
            # Only append the line if it's not already in the melody list
            if line not in songs_dict[current_song_id]["melody"]:
                songs_dict[current_song_id]["melody"].append(line)
    
    return songs_dict

def save_processed_data(lines, file_path):
    """
    Saves the standardized data to a file.
    """
    with open(file_path, "w") as file:
        file.write("\n".join(lines))

def save_lookup_table(songs_dict, file_path):
    """
    Saves the parsed songs dictionary to a JSON file.
    """
    with open(file_path, "w") as file:
        json.dump(songs_dict, file, indent=4)

def preprocess_dataset():
    """
    The main function that ties everything together.
    """
    try:
        print("Loading raw data...")
        raw_lines = load_raw_data(RAW_DATA_PATH)
        
        print("Standardizing formatting...")
        standardized_lines = standardize_formatting(raw_lines)
        
        print("Saving standardized data...")
        save_processed_data(standardized_lines, PROCESSED_DATA_PATH)
        
        print("Parsing data into dictionary...")
        songs_dict = parse_abc_data(standardized_lines)

        print("Saving lookup table...")
        save_lookup_table(songs_dict, LOOKUP_TABLE_PATH)
        
        print("Preprocessing complete!")
        print(f"Standardized data saved to: {PROCESSED_DATA_PATH}")
        print(f"Lookup table saved to: {LOOKUP_TABLE_PATH}")
    
    except Exception as e:
        print(f"Uh-oh! Something went wrong: {e}")

if __name__ == "__main__":
    preprocess_dataset()