from cv2_images import CV2Image, cv2image_from_file, cv2image_from_pil, \
                            pil_to_cv2, cv2_to_pil
import os
import pandas as pd
import numpy as np
from typing import Union, Tuple, List
import torch
from PIL.Image import Image
from PIL.ImageDraw import Draw
from PIL import ImageFont

from rdkit import Chem
from rdkit.Chem.Draw import MolToImage

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import warnings
warnings.simplefilter(action='ignore')
from tqdm import tqdm
import re
from scipy import stats
import multiprocessing as mp
import time
import multiprocessing 
#https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
#to hack daemonic processes https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
import pytesseract

from sys import argv


def refine_subs(string):
    string = re.sub(r'\d{1,2}%', ' ', string)
    string = re.sub(r'\d{1,3}\s*°C', ' ', string)
    string = re.sub(r'\d{1,2}\s*oC', ' ', string)
    string = re.sub(r'[a-zA-Z]{1,2}\s?=\s?\w{1,4}', ' ', string) #'R = SMe, 3g, 73%'
    string = re.sub(r'\d{3,20}', ' ', string)
    string = re.sub(r'\w{7,20}', ' ', string)
    string = re.sub(r'[a-zA-Z]{7,20}', ' ', string)
    string = string.replace("′", "'")
    string = string.replace("’", "'")
    string = string.replace('，', ', ')
    string = string.replace('；', '; ')
    string = string.replace(',', '').replace(';', '').replace(':', '').replace(' ', '').replace('%', '').replace('\\', '').\
                    replace('+', '').replace('()', '')
    
    if string.startswith('(') and string.endswith(')'):
        string = string[1:-1]
    
    return string

def refine_molecule(smiles):
    #if multiple molecules - take with maximum number of atoms
    try:
        if smiles != smiles or smiles is None:
            return None
        if '.' in smiles:
            mols = [Chem.MolFromSmiles(x) for x in smiles.split('.')]
            mol_atoms = {mol: mol.GetNumAtoms() for mol in mols}
            final_mol = max(mol_atoms.items(), key = lambda x: x[1])[0]
            smiles = Chem.MolToSmiles(final_mol)
        if '*' not in smiles:
            return smiles
    except Exception as e:
        return None

def link_molecules_and_subsripts(mols, subs, dist_treshold = 700):
    '''Finds pairs of corresponding molecules and subscripts'''
    # add absent columns
    if 'text_all' not in mols.columns:
        mols.insert(0, 'text_all', '')
    if 'text_below' not in mols.columns:
        mols.insert(1, 'text_below', '')
    if 'sub_id_all' not in mols.columns:
        mols.insert(2, 'sub_id_all', np.nan)
    if 'sub_id_below' not in mols.columns:
        mols.insert(3, 'sub_id_below', np.nan)
    if 'dist_to_sub_all' not in mols.columns:
        mols.insert(4, 'dist_to_sub_all', np.nan)
    if 'dist_to_sub_below' not in mols.columns:
        mols.insert(5, 'dist_to_sub_below', np.nan)
        
    # get closest subscript from below and near. 
    for idx_mol in mols.index:
        dists_below = []
        dists_total = []
        for idx_sub in subs.index:
            dist = np.array(mols.loc[idx_mol, ['xcenter','ycenter']]) - \
                         np.array(subs.loc[idx_sub, ['xcenter','ycenter']])
            dist = (dist**2).sum()**0.5
            if dist > dist_treshold:
                continue
            if subs.loc[idx_sub, 'ycenter'] > mols.loc[idx_mol, 'ycenter']: #подпись ниже чем молекула
                dists_below.append( (dist, idx_sub) )
            dists_total.append( (dist, idx_sub) )
        if len(dists_total) > 0:
            idx_sub_total = min(dists_total, key = lambda x: x[0])[1]
            mols.loc[idx_mol, 'sub_id_all'] = idx_sub_total
            mols.loc[idx_mol, 'text_all'] = subs.loc[idx_sub_total, 'text']
            mols.loc[idx_mol, 'dist_to_sub_all'] = min(dists_total, key = lambda x: x[0])[0]
        if len(dists_below) > 0:
            idx_sub_below = min(dists_below, key = lambda x: x[0])[1] 
            mols.loc[idx_mol, 'sub_id_below'] = idx_sub_below
            mols.loc[idx_mol, 'text_below'] = subs.loc[idx_sub_below, 'text']
            mols.loc[idx_mol, 'dist_to_sub_below'] = min(dists_below, key = lambda x: x[0])[0]
    return mols

def choose_sub(text_below, text_all, sub_id_all, dist_to_below, dist_to_all, sub_id_below_lst):
    if len(text_below) > 0:
        return text_below, dist_to_below
    if sub_id_all not in sub_id_below_lst and len(text_all) > 0:
        return text_all, dist_to_all
    return None, None

def recognize_text(image):
    '''Tesseract wrapper''' # HINT: https://muthu.co/all-tesseract-ocr-options/
    return pytesseract.image_to_string(image, config = '--psm 6')

def recognize_subscripts(init_image, subs):
    '''Recognizes detected subscripts'''
    if subs.empty:
        return
    # get images
    image = init_image.image
    cols = ['x1','y1','x2','y2']
    images = [pil_to_cv2(image.crop(subs.loc[i,cols])) for i in subs.index]
    # ocsr
    subs['text'] = pd.DataFrame({'text': [recognize_text(_) for _ in images]})
    subs['text'] = subs['text'].apply(lambda x: x.replace('\n', ' ').strip())
    return subs

global_folder = argv[1] 

mol_df = pd.read_csv(os.path.join(global_folder, 'df_mols.csv'), sep = '\t')
sub_df = pd.read_csv(os.path.join(global_folder, 'df_subs.csv'), sep = '\t')

image_pathes = list(set(mol_df['image_path']))

def local_recognition(image_path, mol_df=mol_df, sub_df=sub_df):
    print('curr idx', image_pathes.index(image_path))
    try:
        image_path_local = image_path.replace('/home/dchusov/temp_parsing_data', '/media/oleg/second_ssd/rxn_parsing_data_joc')
        
        image = cv2image_from_file(image_path_local)      
        
        mols = mol_df[mol_df['image_path'] == image_path].copy()
        subs = sub_df[sub_df['image_path'] == image_path].copy()
        
        mols.reset_index(inplace = True)
        subs.reset_index(inplace = True)

        if mols.shape[0] == 0 or subs.shape[0] == 0:
            return pd.DataFrame()

        subs = recognize_subscripts(image, subs)
        subs['text'] = subs['text'].apply(refine_subs)
        
        if mols.shape[0] == 0 or subs.shape[0] == 0:
            return pd.DataFrame()
        mols = link_molecules_and_subsripts(mols, subs)
        if mols.shape[0] == 0:
            return pd.DataFrame()
        #подпись по умолчанию взять из text_below. 
        #Если он отсутствует - взять из text_all, но при условии, что этот id нигде больше не заюзан среди text_below
        sub_id_below_lst = [x for x in mols['sub_id_below'].to_list() if (x==x and x is not None)]

        mols['text'], mols['dist_to_sub'] = zip(*mols.apply(lambda row: choose_sub(row['text_below'], 
                                                                                 row['text_all'], 
                                                                                 row['sub_id_all'], 
                                                                                 row['dist_to_sub_below'],
                                                                                 row['dist_to_sub_all'], 
                                                                                 sub_id_below_lst), axis = 1))
        mols.drop(['text_all', 'text_below', 'sub_id_all', 'sub_id_below',
                   'dist_to_sub_all', 'dist_to_sub_below', 'recognition_confidence', 'x1', 'y1', 'x2', 'y2', 
                   'xcenter', 'ycenter',
                   'detection_confidence'], axis = 1, inplace = True)
        mols.dropna(subset = 'dist_to_sub', inplace = True)

        #выпадающие значения по расстоянию
        if mols.shape[0]>5:
            mols = mols[(np.abs(stats.zscore(mols['dist_to_sub'])) < 3)].copy()

        mols['smiles'] = mols['smiles'].apply(refine_molecule)
        mols.dropna(inplace = True)
        mols['file_path'] = [image_path]*mols.shape[0]

        if mols.shape[0] == 0 :
            return pd.DataFrame()
        return mols
    except Exception as e:
        print('ERROR \n\n', e, '\n', image_path, '\n\n')    
        return pd.DataFrame()
    
print('start time: ', time.time())
with mp.Pool(80) as p:
    mol_df_lst = p.map(local_recognition, image_pathes)  
print('end time: ', time.time())
print(len(mol_df_lst))    
# mol_df_lst = []
# for image_path in tqdm(image_pathes):
#     mols = local_recognition(image_path)
#     mol_df_lst.append(mols)
    
    
#     try:
#         image_path_local = image_path.replace('/home/dchusov/temp_parsing_data', '/media/oleg/second_ssd/rxn_parsing_data')
        
#         image = cv2image_from_file(image_path_local)      
        
#         mols = mol_df[mol_df['image_path'] == image_path].copy()
#         subs = sub_df[sub_df['image_path'] == image_path].copy()
        
#         mols.reset_index(inplace = True)
#         subs.reset_index(inplace = True)

#         if mols.shape[0] == 0 or subs.shape[0] == 0:
#             continue

#         subs = recognize_subscripts(image, subs)
#         subs['text'] = subs['text'].apply(refine_subs)
        
#         if mols.shape[0] == 0 or subs.shape[0] == 0:
#             continue
#         mols = link_molecules_and_subsripts(mols, subs)
#         if mols.shape[0] == 0:
#             continue
#         #подпись по умолчанию взять из text_below. 
#         #Если он отсутствует - взять из text_all, но при условии, что этот id нигде больше не заюзан среди text_below
#         sub_id_below_lst = [x for x in mols['sub_id_below'].to_list() if (x==x and x is not None)]

#         mols['text'], mols['dist_to_sub'] = zip(*mols.apply(lambda row: choose_sub(row['text_below'], 
#                                                                                  row['text_all'], 
#                                                                                  row['sub_id_all'], 
#                                                                                  row['dist_to_sub_below'],
#                                                                                  row['dist_to_sub_all'], 
#                                                                                  sub_id_below_lst), axis = 1))
#         mols.drop(['text_all', 'text_below', 'sub_id_all', 'sub_id_below',
#                    'dist_to_sub_all', 'dist_to_sub_below', 'recognition_confidence', 'x1', 'y1', 'x2', 'y2', 
#                    'xcenter', 'ycenter',
#                    'detection_confidence'], axis = 1, inplace = True)
#         mols.dropna(subset = 'dist_to_sub', inplace = True)

#         #выпадающие значения по расстоянию
#         if mols.shape[0]>5:
#             mols = mols[(np.abs(stats.zscore(mols['dist_to_sub'])) < 3)].copy()

#         mols['smiles'] = mols['smiles'].apply(refine_molecule)
#         mols.dropna(inplace = True)
#         mols['file_path'] = [image_path]*mols.shape[0]

#         if mols.shape[0] == 0 :
#             continue
#         mol_df_lst.append(mols)
#     except Exception as e:
#         print('ERROR \n\n', e, '\n', image_path, '\n\n')
        
if len(mol_df_lst) > 0:
    mol_df_final = pd.concat(mol_df_lst)
else:
    mol_df_final = pd.DataFrame()

mol_df_final.to_csv(os.path.join(global_folder, 'df_subs_mols.csv'), index = False, sep = '\t')
