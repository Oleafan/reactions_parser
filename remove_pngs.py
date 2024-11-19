import os
import json
from sys import argv

global_folder = argv[1]

for folder in os.listdir(global_folder):
    if not os.path.isdir(os.path.join(global_folder, folder)):
        continue
    for file in os.listdir(os.path.join(global_folder, folder)):
        if file.endswith('.png'):
            os.remove(os.path.join(global_folder, folder, file))
print('files deleted')