import os
import json
from sys import argv

global_folder = argv[1]

parse_id = global_folder.split('/')[-1]
#copy_protocols with splitting by 5000 reactions

def create_sbatch(parse_id, num_parts):
    string = f"""#! /bin/bash
#SBATCH --job-name="{parse_id + '_' + str(num_parts)}"
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-24:0
#SBATCH --output="stdout.%j.txt"%j.out
#SBATCH --error="stderr.%j.txt"%j.out
module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate mistral

nvidia-smi

global_folder="/home/dchusov/mistral_temp/{parse_id}/{str(parse_id) + '_' + str(num_parts)}" 
python eval_mistral_multibatch_multifolder.py $global_folder"""
    return string
    
    
    

num_parts = 1
dataset = []
for folder in os.listdir(global_folder):
    if not os.path.isdir(os.path.join(global_folder, folder)):
        continue
    with open(os.path.join(global_folder, folder, 'protocols.json')) as f:
        local_dataset = json.loads(f.read())
#         print(folder, ':', len(local_dataset), 'potential protocols')
    for x in local_dataset:
        x['init_paper'] = folder
        
    dataset += local_dataset
    
    if len(dataset) > 3000:
        parse_folder = os.path.join(global_folder, str(parse_id) + '_' + str(num_parts))
        os.mkdir(parse_folder)
        with open(os.path.join(parse_folder, 'init_protocols' + '.json'), 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(dataset))
        with open(os.path.join(parse_folder, parse_id + '_' + str(num_parts) + '.sbatch'), 'w', encoding = 'utf-8') as f:
            f.write(create_sbatch(parse_id, num_parts))
        dataset = []
        num_parts += 1
    
parse_folder = os.path.join(global_folder, str(parse_id) + '_' + str(num_parts))
os.mkdir(parse_folder)
    
with open(os.path.join(parse_folder, 'init_protocols.json'), 'w', encoding = 'utf-8') as f:
    f.write(json.dumps(dataset))
with open(os.path.join(parse_folder, parse_id + '_' + str(num_parts) + '.sbatch'), 'w', encoding = 'utf-8') as f:
    f.write(create_sbatch(parse_id, num_parts))
