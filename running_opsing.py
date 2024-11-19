import subprocess
import os
from sys import argv
import re
import time

temp_folder = argv[1]
print('running opsin in', temp_folder)

OPSIN = '/media/oleg/second_ssd/reaction_parser/opsin280.jar'

def name_to_smiles(filename_input, filename_output, OPSIN = OPSIN):
    command = 'java -jar ' + OPSIN + ' -s -osmi ' + filename_input + ' ' + filename_output
    call = subprocess.run(command, shell=True, text=True, capture_output=True)

processed_folders = []

while True:
    if len(os.listdir(temp_folder)) == 0:
        time.sleep(60)
        continue
    print('number of folders: ', len(os.listdir(temp_folder)))
    for subfolder in os.listdir(temp_folder):
        subfolder_path = os.path.join(temp_folder, subfolder)
        if os.path.isdir(subfolder_path):
            if (os.path.exists(os.path.join(subfolder_path, 'marker_file.txt')) and 
                subfolder_path not in processed_folders): #creation of text and pngs finished
                #check if smileses were already generated
                if not os.path.exists(os.path.join(subfolder_path, 'smiles.txt')):

                    filename_input = os.path.join(subfolder_path, 'candidates.txt')
                    filename_output = os.path.join(subfolder_path, 'smiles.txt')
                    name_to_smiles(filename_input, filename_output)
                
                processed_folders.append(subfolder_path)
                print(subfolder_path, 'opsin done')                
    
    total_folders = [os.path.join(temp_folder, x) for x in os.listdir(temp_folder)]
    total_folders = [x for x in total_folders if os.path.isdir(x)]

    if set(total_folders) - set(processed_folders) == set():
        break

    time.sleep(20)
    