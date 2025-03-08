import os
import json
import re
from collections import defaultdict

# Define paths for the dataset and output files
RAW_DATA_PATH = "data/raw/abc_notations.txt"  # Path to the raw ABC notation file
PROCESSED_DATA_PATH = "data/processed/standardized_abc.txt"  # Path to save the standardized data
LOOKUP_TABLE_PATH = "data/lookup_tables/songs_dict.json"  # Path to save the parsed songs dictionary

# Create directories if they don't exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOOKUP_TABLE_PATH), exist_ok=True)

# Tokens to mark the start and end of annotations
START_ANN_TOKEN = "<START_ANN>"
END_ANN_TOKEN = "<END_ANN>"

def load_raw_data(file_path):
    """
    Loads the raw ABC notation data from the specified file.
    I added a check to make sure the file exists—nothing worse than running a script
    and realizing the data isn't there!
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Oops! The file {file_path} doesn't exist. Double-check the path.")
    
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    return lines

def standardize_formatting(lines):
    """
    Cleans up the raw ABC notation lines and adds tokens to annotations.
    This step is crucial because the raw data can be messy—extra spaces, inconsistent
    line breaks, and annotations scattered everywhere. I'm adding special tokens to
    make annotations easier to parse later.
    """
    standardized_lines = []
    for line in lines:
        line = line.strip()  # Get rid of any extra spaces at the start or end
        if not line:  # Skip empty lines—they just take up space
            continue
        
        # If the line is an annotation (starts with '%'), wrap it in tokens
        if line.startswith("%"):
            line = f"{START_ANN_TOKEN} {line} {END_ANN_TOKEN}"
        
        standardized_lines.append(line)
    
    return standardized_lines

def parse_abc_data(lines):
    """
    Parses the standardized ABC notation into a structured dictionary.
    Each song gets its own entry with fields like 'id', 'title', 'time_signature',
    'note_length', and 'annotations'. I'm using a defaultdict to handle missing
    fields gracefully—no crashes if something's not there!
    """
    songs_dict = defaultdict(dict)
    current_song_id = None
    
    for line in lines:
        # Extract the song ID (lines starting with 'X:')
        if line.startswith("X:"):
            current_song_id = line.split(":")[1].strip()
            songs_dict[current_song_id]["id"] = current_song_id
        
        # Extract the title (lines starting with 'T:')
        elif line.startswith("T:"):
            songs_dict[current_song_id]["title"] = line.split(":")[1].strip()
        
        # Extract the time signature (lines starting with 'M:')
        elif line.startswith("M:"):
            songs_dict[current_song_id]["time_signature"] = line.split(":")[1].strip()
        
        # Extract the note length (lines starting with 'L:')
        elif line.startswith("L:"):
            songs_dict[current_song_id]["note_length"] = line.split(":")[1].strip()
        
        # Extract annotations (lines with <START_ANN> and <END_ANN> tokens)
        elif START_ANN_TOKEN in line:
            # Remove the tokens and clean up the annotation
            annotation = re.sub(f"{START_ANN_TOKEN}|{END_ANN_TOKEN}", "", line).strip()
            if "annotations" not in songs_dict[current_song_id]:
                songs_dict[current_song_id]["annotations"] = []
            songs_dict[current_song_id]["annotations"].append(annotation)
    
    return songs_dict

def save_processed_data(lines, file_path):
    """
    Saves the standardized data to a file.
    This is just a simple function to write the cleaned-up lines to a new file.
    """
    with open(file_path, "w") as file:
        file.write("\n".join(lines))

def save_lookup_table(songs_dict, file_path):
    """
    Saves the parsed songs dictionary to a JSON file.
    I'm using json.dump with indentation to make the file human-readable—because
    nobody likes squinting at a wall of text!
    """
    with open(file_path, "w") as file:
        json.dump(songs_dict, file, indent=4)

def preprocess_dataset():
    """
    The main function that ties everything together.
    This is where the magic happens! It loads the raw data, standardizes it, parses
    it into a dictionary, and saves the results. I added some print statements to
    track progress—it's always nice to see what's happening as the script runs.
    """
    try:
        # Step 1: Load the raw data
        print("Loading raw data...")
        raw_lines = load_raw_data(RAW_DATA_PATH)
        
        # Step 2: Standardize the formatting
        print("Standardizing formatting...")
        standardized_lines = standardize_formatting(raw_lines)
        
        # Step 3: Save the standardized data
        print("Saving standardized data...")
        save_processed_data(standardized_lines, PROCESSED_DATA_PATH)
        
        # Step 4: Parse the data into a dictionary
        print("Parsing data into dictionary...")
        songs_dict = parse_abc_data(standardized_lines)
        
        # Step 5: Save the lookup table
        print("Saving lookup table...")
        save_lookup_table(songs_dict, LOOKUP_TABLE_PATH)
        
        print("Preprocessing complete!")
        print(f"Standardized data saved to: {PROCESSED_DATA_PATH}")
        print(f"Lookup table saved to: {LOOKUP_TABLE_PATH}")
    
    except Exception as e:
        print(f"Uh-oh! Something went wrong: {e}")

if __name__ == "__main__":
    preprocess_dataset()