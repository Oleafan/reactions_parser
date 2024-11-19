import pdf2text as p2t
import os
import json
from pathlib import Path
from sys import argv

folder_path = argv[1] #folder with initial data (pdfs)
temp_dir = argv[2] #'/media/oleg/second_ssd/temp_files_parser'
file_list = [os.path.join(folder_path, x) for x in os.listdir(folder_path)]

#file_list = ['/media/oleg/second_ssd/pdf_for_parse/ol3c01992_si_001.pdf']
#doi = '1001/ol3c01992'

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

