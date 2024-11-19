import json 
from sys import argv
import os

from rdkit import Chem
from indigo import Indigo
indigo = Indigo()
import re

import pandas as pd

import reactions_converter as rc

import copy
from itertools import permutations, combinations  


main_folder = argv[1]
import warnings
warnings.filterwarnings('ignore')
# global_folder = '/media/oleg/second_ssd/temp_files_parser'

for parse_id in os.listdir(main_folder):
    if not os.path.isdir(os.path.join(main_folder, parse_id)):
        continue
    if not os.path.exists(os.path.join(main_folder, parse_id, 'reactions_parsed.json')):
        continue
        
    global_folder = os.path.join(main_folder, parse_id)
    
    # folder_name = '10.1039_c8ob02305k'
    for folder_name in os.listdir(global_folder):
        folder = os.path.join(global_folder, folder_name)
        if not os.path.isdir(folder):
            continue
        

        with open(os.path.join(folder, 'joint_sorted_dict.json')) as f:
            joint_sorted_dict = json.loads(f.read())
        with open(os.path.join(folder, 'compid_descr.json')) as f:
            compid_descr = json.loads(f.read()) 
        with open(os.path.join(folder, 'code2compid.json')) as f:
            code2compid = json.loads(f.read()) 
        with open(os.path.join(folder, 'compid2name.json')) as f:
            compid_to_name = json.loads(f.read()) 
        with open(os.path.join(global_folder, 'reactions_parsed.json')) as f:
            protocols = json.loads(f.read())
        protocols = [x for x in protocols if x['init_paper'] == folder_name]
        
        transformed_protocols = []
        for protocol in protocols:
            try:
                transformed_protocols += rc._transform_reaction_dict(protocol) #extract reagents names and reorganize naming
            except Exception as e:
                pass #print(e, '\n', protocol)
        for idx, item in enumerate(transformed_protocols):
            item['is_general'] = rc._is_general(item['procedure'], item['procedure_name'])
            item['id'] = idx
        gen_prot_dict = {}
        for item in transformed_protocols:
            item['refers_to'] = []
            if item['is_general']:
                for name in list(set(item['procedure_name'])):
                    gen_prot_dict[name] = item['id']
            else:
                for gp_name in gen_prot_dict:
                    if gp_name.replace(' ', '') in item['procedure'].replace(' ', ''):
                        item['refers_to'].append(gen_prot_dict[gp_name])

        reactions = []
        gen_proc_cond = {} # {idx: {'conditions': conditions, 'protocol': procedure}}

        used_idx = []
        for reaction in transformed_protocols:
            reaction_created = False

            if reaction['is_general']:
                #в этом случае из протокола сохраняем отдельно условия, MiscInch и растворители
                misc_solv_list = []
                for comp in reaction['compounds']:
                    if comp['compound_role'] != 'target': 
                        misc_solv_list.append(comp)

                gp_ent = {'procedure': reaction['procedure'],
                          'conditions': reaction['conditions'],
                          'misc_solv_list': misc_solv_list } #=solv_gen
                gen_proc_cond[reaction['id']] = gp_ent

            elif len(reaction['refers_to']) > 0:
                #дополняем реакцию условиями из общей методики
                for gen_proc_id in reaction['refers_to']:
                    reaction['conditions'] += gen_proc_cond[gen_proc_id]['conditions']

                    #check if there are compounds in gen_proc duplicate already mentioned compounds in reaction
                    existing_comps_with_amounts = []
                    for comp in reaction['compounds']:
                        if len(comp['amounts']) > 0:
                            existing_comps_with_amounts.append(comp['compound_id'])
                    appending_comps = []
                    for comp in gen_proc_cond[gen_proc_id]['misc_solv_list']:
                        if comp['compound_id'] not in existing_comps_with_amounts:
                            appending_comps.append(comp)

                    reaction['compounds'] = appending_comps + reaction['compounds'] #ставим общие условия в начале            
            reaction_smi = rc._transform_reaction_combs(reaction, compid_descr, code2compid, compid_to_name)

            reaction_smi['initial_protocol'] = reaction['procedure']
            reaction_smi['used_idx'] = reaction['idx'] if type(reaction['idx']) == list else [reaction['idx']]
            reactions.append(reaction_smi)

        used_idx = []
        final_rxn_list = []
        for reaction in reactions:
            if rc._is_true_reaction(reaction):
                already_added = False
                for idx in reaction['used_idx']:
                    if idx in used_idx:
                        already_added = True

                if not already_added:
                    final_rxn_list.append(reaction)
                    used_idx += reaction['used_idx']

        for reaction in reactions:
            if rc._is_true_product(reaction):
                already_added = False
                for idx in reaction['used_idx']:
                    if idx in used_idx:
                        already_added = True

                if not already_added:
                    final_rxn_list.append(reaction)
                    used_idx += reaction['used_idx']
        for rxn in reactions:
            rxn.pop('rxn', None) 
        print(folder_name, ':', len(final_rxn_list), 'reactions')
        with open(os.path.join(folder, 'reactions_final.json'), 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(final_rxn_list))
#Find reactions created from all protocols with check that one idx should be used only once. Next check if there are reactions with only product.  