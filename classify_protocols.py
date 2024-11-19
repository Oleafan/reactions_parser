from sys import argv
import json
from token_classifier.bert_classifier import get_classes
import re
import os

folder = argv[1]

with open(os.path.join(folder, 'splitted_tokens.json')) as f:
    splitted_tokens = json.loads(f.read())

tokens_with_classes = [{'id': idx, 'token': token, 'class': -1} for idx, token in enumerate(splitted_tokens) ]

regexes_yield = [r'[Pp]urified', r'[Ii]solated', r'obtain', r'[Pp]ure', 'desired', 
                 '[Pp]rovided', '[Yy]ield', '[Aa]fford', '[Ff]urnish']
    
for item in tokens_with_classes:
    if len(item['token']) < 150:
        for regex in regexes_yield:
            if re.search(regex, item['token']):
                item['class']  = 1
                break
    elif len(item['token']) > 1500:
        item['class'] = 0    
    if '............' in item['token']:
        item['class'] = 0
        
tokens_for_bert = []
for item in tokens_with_classes:
    if item['class'] == -1:
        tokens_for_bert.append(item['token'])
labels_bert = get_classes(tokens_for_bert) 
for item, label in zip([x for x in tokens_with_classes if x['class'] == -1], labels_bert):
    item['class'] = label
    
#фильтруем только протоколы и выставляем дистанцию перед ними
protocols = []
dist_before = 0
for item in tokens_with_classes:
    if item['class'] == 1:
        temp_dict = {'protocol': item['token'],
                    'dist_before': dist_before}
        protocols.append(temp_dict)
        dist_before = 0
    else:
        dist_before += len(item['token'])
print('Protocols found: ', len(protocols))

with open(os.path.join(folder, 'protocols.json'), 'w', encoding = 'utf-8') as f:
    f.write(json.dumps(protocols))