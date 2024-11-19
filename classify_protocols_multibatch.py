from sys import argv
import json
from token_classifier.bert_classifier import get_classes
import re
import os
import warnings
warnings.filterwarnings("ignore")

global_folder = argv[1]

def _join_protocols(protocol_dict_1, protocol_dict_2):
    #объединяет два протокола
    new_protocol = ' '.join([protocol_dict_1['protocol'], protocol_dict_2['protocol']])
    dist_before = protocol_dict_1['dist_before']
    return {'protocol': new_protocol,
            'dist_before': dist_before,
            'idx': [protocol_dict_1['idx'], protocol_dict_2['idx']]}

for folder_name in os.listdir(global_folder):
    
    folder = os.path.join(global_folder, folder_name)
    if not os.path.isdir(folder):
        continue
        
    if os.path.exists(os.path.join(folder, 'protocols.json')):
        continue
    
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
    print(folder_name, 'initially protocols found: ', len(protocols))
    
    #numerate protocols:
    for idx, protocol in enumerate(protocols):
        protocol['idx'] = idx
    
    #join protocols
    new_protocol_list = []
    for idx, protocol in enumerate(protocols):
        new_protocol_list.append(protocol)
        if (idx > 0 and protocol['dist_before'] < 100):
            new_protocol_list.append(_join_protocols(protocols[idx-1], protocol))
    print(folder_name, 'protocols after join found: ', len(new_protocol_list))
    
    with open(os.path.join(folder, 'protocols.json'), 'w', encoding = 'utf-8') as f:
        f.write(json.dumps(new_protocol_list))