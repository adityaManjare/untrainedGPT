import lzma 
import os
from tqdm import tqdm

def xz_files_in_directory(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.xz') and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = "D:/llm/openwebtext/openwebtext"
output_file_val = "output_val.txt"
output_file_train = "output_train.txt"
vocab_file = "vocab.txt"


files = xz_files_in_directory(folder_path)

total_files = len(files)
split_index=int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

# processing the training files

with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text )
            characters = set(text)
            vocab.update(characters)
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text )
            characters = set(text)
            vocab.update(characters)

with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')
