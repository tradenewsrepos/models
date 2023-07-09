import os
import shutil
with open('golden_files.txt', 'r') as file:
    golden_files = file.readlines()
    golden_files = [f.replace('\n', '') for f in golden_files if '.ann' in f]

golden_files = [f.replace('brat/data/relations', '') for f in golden_files]
DATA_PATH = "../data/test_models/NER_RE_silver_data/"
print(len(golden_files))

for f in golden_files:
    file_path = DATA_PATH + f
    file_path_text = file_path.replace('ann', 'txt')
    if os.path.exists(file_path):
        shutil.copy(file_path, '../data/test_models/NER_RE_golden_files')
    if os.path.exists(file_path_text):
        shutil.copy(file_path_text, '../data/test_models/NER_RE_golden_files')