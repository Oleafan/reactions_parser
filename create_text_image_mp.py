import pdf2text as p2t
import os
import json
from pathlib import Path
from sys import argv
import multiprocessing as mp

global_folder = argv[1] #folder with initial data (folders with pdfs)
temp_dir = argv[2] #'/media/oleg/second_ssd/temp_files_parser'

list_of_papers_file = temp_dir + '_input.json'

def create_text_images(folder_path):
    file_list = [os.path.join(folder_path, x) for x in os.listdir(folder_path)]
    subfolder_name = folder_path.split('/')[-1]
    temp_folder = os.path.join(temp_dir, subfolder_name)
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    text = p2t.pdf2text_image(file_list, temp_folder)
    clean_text = p2t.clean_exp_sec(text)
    tokens = p2t.tokenize_exp_sec(clean_text)

    with open(os.path.join(temp_folder, 'tokens.json'),  'w', encoding = 'utf-8') as f:
        f.write(json.dumps(tokens))

    candidates = p2t.get_candidates_whole(tokens)

    with open(os.path.join(temp_folder, 'candidates.txt'), 'w', encoding="utf-8") as fp:
        for item in candidates:
            # write each item on a new line
            fp.write("%s\n" % item)
            
    with open(os.path.join(temp_folder, 'marker_file.txt'), 'w') as fp:
        fp.write('done')
#     print('Text and images created for', folder_path)
    return None


# folders_list = [os.path.join(global_folder, x) for x in os.listdir(global_folder)]
with open(list_of_papers_file) as f:
    folders_list = json.loads(f.read()) #just list of folders, not a full path)  

folders_list = [os.path.join(global_folder, x) for x in folders_list]
folders_list = [x for x in folders_list if os.path.isdir(x)]

print('number of files:', len(folders_list ) )

with mp.Pool(16) as p:
    p.map(create_text_images, folders_list) 
