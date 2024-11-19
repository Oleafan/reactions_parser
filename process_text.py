import json
import os
from sys import argv
import pdf2text as p2t
import pandas as pd

folder = argv[1] #'/media/oleg/second_ssd/temp_files_parser/1001_ol3c01992'
with open(os.path.join(folder, 'tokens.json')) as f:
    tokens = json.loads(f.read())
    
name_to_smiles, solv_dict, misc_dict = p2t.get_name_to_smi(folder)
spec_list = p2t.get_spec_list(tokens, name_to_smiles)

exp_sec_new = ' '.join(tokens)
exp_sec_new = p2t.segmentate_exp_sec(exp_sec_new, spec_list)

name_to_smiles = p2t.remove_extra_smiles(name_to_smiles)
misc_dict = p2t.remove_extra_smiles(misc_dict)

joint_sorted_dict, compid_descr = p2t.get_joint_substr_dict(name_to_smiles, solv_dict, misc_dict)

try:
    code_df = pd.read_csv(os.path.join(folder, 'total.csv'), sep = '\t') #from ocr
except Exception as e:
    code_df = pd.DataFrame(columns = ['text_tesseract','sub_id','detection_confidence','file','smiles',
                                      'recognition_confidence','dist_to_sub'])


code2compid, compid_descr, exp_sec_new = p2t.create_code2compid(exp_sec_new, joint_sorted_dict, compid_descr, code_df)

token_list = p2t.final_text_processing(exp_sec_new)

#sorting of dicts
compid_to_name = {v:k for k,v in joint_sorted_dict.items()} #словарь вида {'Compound_35: 'IUPAC NAME'}
keys = sorted(compid_to_name.keys(), key = len, reverse = True)
compid_to_name = {k: compid_to_name[k] for k in keys} 

#словаь вида {'Compound_25': SMILES}, отсортированный по убыыванию длины ключа
sorted_keys = sorted(compid_descr.keys(), key=len, reverse = True)
compid_descr = dict(zip(sorted_keys, [compid_descr[key] for key in sorted_keys]))  

#расшифровка кодов компаундов '1a', '1' и т.п. вида {'1a': 'Compound_25'}. 
sorted_keys = sorted(code2compid.keys(), key=len, reverse = True)
code2compid = dict(zip(sorted_keys, [code2compid[key] for key in sorted_keys]))


with open(os.path.join(folder, 'joint_sorted_dict.json'), 'w') as f:
    f.write(json.dumps(joint_sorted_dict))
with open(os.path.join(folder, 'compid2name.json'), 'w') as f:
    f.write(json.dumps(compid_to_name))
with open(os.path.join(folder, 'compid_descr.json'), 'w') as f:
    f.write(json.dumps(compid_descr))
with open(os.path.join(folder, 'text.txt'), 'w') as f:
    f.write(exp_sec_new)
with open(os.path.join(folder, 'code2compid.json'), 'w') as f:
    f.write(json.dumps(code2compid))
with open(os.path.join(folder, 'splitted_tokens.json'), 'w') as f:
    f.write(json.dumps(token_list))