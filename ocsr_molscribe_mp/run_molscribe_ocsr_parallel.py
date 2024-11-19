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

import warnings
warnings.simplefilter(action='ignore')
from tqdm import tqdm
import re
from scipy import stats
# import multiprocessing as mp

import multiprocessing
#https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
#to hack daemonic processes https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
import pytesseract
from molscribe import MolScribe
from ultralytics import YOLO

from sys import argv

def refine_mol_boundaries(df: pd.DataFrame, delta: int = 5) -> None:
    '''Refines boundaries of molecule image boundaries'''
    # TODO: improve algorithm
    df['x1'] -= delta
    df['y1'] -= delta
    df['x2'] += delta
    df['y2'] += delta
    return None


def refine_subscript_boundaries(df: pd.DataFrame, delta: int = 5) -> None:
    '''Refines boundaries of subscription image boundaries'''
    # TODO: improve algorithm
    df['x1'] -= delta
    df['y1'] -= delta
    df['x2'] += delta
    df['y2'] += delta
    return None

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

def ocsr_superfunction(image_path_list):

    
    def recognize_text(image):
        '''Tesseract wrapper''' # HINT: https://muthu.co/all-tesseract-ocr-options/
        return pytesseract.image_to_string(image, config = '--psm 6')
    
    #MODELS
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_MOLSCRIBE = MolScribe('/media/oleg/second_ssd/models_for_odanreact_molscribe/molscribe_swin_base_char_aux_1m680k.pth', DEVICE)
    MODEL_DETECT = YOLO('/media/oleg/second_ssd/models_for_odanreact_molscribe/moldetector_yolo-v8.pt')
    
    def recognize_molecules_molscribe(images) :
        '''Returns SMILES and confidence for the given image of molecule'''
        results = MODEL_MOLSCRIBE.predict_images(images, return_confidence = True)
        return results

    def detect_mols_and_subs(image: Union[CV2Image, Image, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''Detects molecules and the corresponding subscriptions on the image'''
        # get recognized objects
        result = MODEL_DETECT(image.image);
        classes = result[0].boxes.data
        df = pd.DataFrame(classes.cpu(), dtype=float,
                          columns=['x1', 'y1', 'x2', 'y2', 'detection_confidence', 'class'])
        df.loc[:,'class'] = df.loc[:,'class'].apply(lambda x: result[0].names[int(x)])
        # check minimaxes
        for ax in 'xy':
            df[[f'{ax}1',f'{ax}2']] = df[[f'{ax}1',f'{ax}2']].apply([min, max], axis = 1)
        # split to two dataframes
        mols = df.loc[df['class'] == 'mol'].reset_index(drop = True)
        mols = mols[mols.columns[:-1]]
        col_idx = list(mols.columns).index('detection_confidence')
        mols.insert(col_idx    , 'xcenter', mols[['x1','x2']].apply('mean', axis = 1))
        mols.insert(col_idx + 1, 'ycenter', mols[['y1','y2']].apply('mean', axis = 1))
        refine_mol_boundaries(mols)
        subs = df.loc[df['class'] == 'desc'].reset_index(drop = True)
        subs = subs[subs.columns[:-1]]
        subs.insert(col_idx    , 'xcenter', subs[['x1','x2']].apply('mean', axis = 1))
        subs.insert(col_idx + 1, 'ycenter', subs[['y1','y2']].apply('mean', axis = 1))
        refine_subscript_boundaries(subs)
        return mols, subs
    
    def detected_objects(image, mols, subs):
        '''Initial image with highlighted detected objects'''
        #for tessting only
        mapped_image = image.image.copy()
        draw = Draw(mapped_image)
        font = ImageFont.load_default(16) # TODO: change to custom
        delta = 5
        for df, color in [(mols, 'blue'), (subs, 'red')]:
            for i in df.index:
                points = list(df.loc[i,['x1','y1','x2','y2']])
                draw.rectangle(points, outline=color, width=2)
                draw.text([_ + delta for _ in points[:2]], str(i), font=font,
                          fill='orange', stroke_width=1, stroke_fill='orange')
        return mapped_image 
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
    
    def recognize_molecules(init_image, mols):
        '''Recognizes detected molecules'''
        if mols.empty:
            return
        # add absent columns
        if 'smiles' not in mols.columns:
            mols.insert(0, 'smiles', '')
        if 'recognition_confidence' not in mols.columns:
            mols.insert(1, 'recognition_confidence', 0.0)
        # ocsr magic
        image = init_image.image
        cols = ['x1','y1','x2','y2']
        images = [cv2image_from_pil(image.crop(mols.loc[i,cols])) for i in mols.index]
        recognized_molecules = recognize_molecules_molscribe(images)
        # add main mol info to the dataframe
        mols['smiles'] = [rm['smiles'] for rm in recognized_molecules]
        mols['recognition_confidence'] = [rm['confidence'] for rm in recognized_molecules]

        return mols
    
    mol_df_lst = []
    for image_path in tqdm(image_path_list):
        try:
            image = cv2image_from_file(image_path)
            mols, subs = detect_mols_and_subs(image)
            if mols.shape[0] == 0 or subs.shape[0] == 0:
                continue
#             detected_objects(image, mols, subs).save(image_path.replace('.png', 'rec.png')) #save this png
            subs = recognize_subscripts(image, subs)
            subs['text'] = subs['text'].apply(refine_subs)
            if mols.shape[0] == 0 or subs.shape[0] == 0:
                continue
            mols = recognize_molecules(image, mols)
            mols = link_molecules_and_subsripts(mols, subs)
            if mols.shape[0] == 0:
                continue
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
                continue
            mol_df_lst.append(mols)
        except Exception as e:
            print('ERROR \n\n', e, image_path, '\n\n')
        
    if len(mol_df_lst) > 0:
        return pd.concat(mol_df_lst)
    else:
        return pd.DataFrame()


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)
        

def check_finish_status(folder):
    status = True
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            if not os.path.exists(os.path.join(subfolder_path, 'marker_file.txt')):
                status = False #if at least one subfolder does not contain this file
    return status
    
global_folder = argv[1]

processed_file_list = []
finish_status = False
res_lst = []

while True:
    total_file_list = []
    for subfolder in os.listdir(global_folder):
        if not os.path.isdir(os.path.join(global_folder, subfolder)):
            continue
        for file in os.listdir(os.path.join(global_folder, subfolder)):
            if file.endswith('.png') and (not file.endswith('rec.png')):
                if os.path.join(global_folder, subfolder, file) not in processed_file_list:
                    total_file_list.append(os.path.join(global_folder, subfolder, file))

    print('Total number of png files:', len(total_file_list))
    
    if len(total_file_list) == 0 and check_finish_status(global_folder):
        print('breaking')
        break

    num_batches = 8
    num_per_batch = int(len(total_file_list)/num_batches)

    batches = []
    for idx in range(num_batches):
        if idx < num_batches-1:
            batches.append(total_file_list[idx*num_per_batch: (idx+1)*num_per_batch])
        else: 
            batches.append(total_file_list[idx*num_per_batch:])

    

    with NestablePool(num_batches) as p:
        result = p.map(ocsr_superfunction, batches)  
        
    processed_file_list += total_file_list
    res_lst += result
    
res_df = pd.concat(res_lst)
res_df.to_csv(os.path.join(global_folder, 'df_subs_mols.csv'), index = False, sep = '\t')
                                   
                                   
            
    
    
    
    
    
    
    
    
    
    
    
    
    
