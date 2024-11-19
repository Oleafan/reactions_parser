import spacy
from spacy import displacy
# nlp = spacy.load("en_core_web_sm")
import random
from spacy.lang.en import English
# from spacy.training.example import Example
# from spacy.util import minibatch, compounding
import re
import pandas as pd
import warnings
import copy
# from IPython.display import clear_output
import json
import scispacy
import chemdataextractor
from chemdataextractor.doc import Sentence
from tqdm import tqdm
warnings.filterwarnings('ignore')

nlp_ner = spacy.load("protocols_parsing/model-best_2709")
nlp_ss = spacy.load("en_core_sci_sm")


targ_refs = ['Prepared', 'prepared', 'Obtained', 'obtained', 'Procedure', 'procedure', 'Preparation', 'preparation', 'Synthesis', 'synthesis']

def posed(raw_text):
    text_to_pos = nlp_ss(raw_text)
    potent_target = ['---']
    sentence = Sentence(raw_text)
    cde_cems = [chem.text for chem in sentence.cems]
    for token in text_to_pos:
        if (token.head.text in targ_refs and ('ompound_' in token.text or token.text in cde_cems)):
            potent_target.append(token.text)
            return token.text
        elif (('ompound_' in token.head.text or token.head.text in cde_cems) and token.text in targ_refs):
            potent_target.append(token.head.text)
            return token.head.text
    return potent_target[0]

amount_marks = ['mg', 'mol', '%', 'µmol', 'mmol%', 'mol%', 'mmol', 'ml', 'mL',
                'µL', 'μL', 'L', 'mA','equiv', 'equiv.', 'eq.', 'M', 'ee', '%ee','wt.', 'µl', 'g']
pattern_1 = re.compile(r"\d+\s+(?:{})".format("|".join(amount_marks)))
pattern_2 = re.compile(r"\d+(?:{})".format("|".join(amount_marks)))

def ner_to_dict(text):
    text_doc = nlp_ner(text)
    targ_from_pos = posed(text)
    entits = [(i, i.label_, i.start_char, i.end_char) for i in text_doc.ents]
    cde_doc = Sentence(text)
    cde_entits = [(chem.text, chem.start, chem.end) for chem in cde_doc.cems]
    cde_ends = [endi for t, s, endi in cde_entits]
    reagents, conditions, yields, procedures, unassigned = [['', '']], [], [], [], []
    reags = ['COMPOUND', 'CHEMICALS', 'MISCINCH', 'SOLVENT']
    conds = ['TIME', 'TEMPERATURE', 'PRESSURE']
    last_mark = ['', 0, '', '', '', 0]
    comp_in_yields_count = 0
    yield_counter = 0
    comps_list = []
    first_reagent = ['', '']
    yields_dict = {} 
    chars_list = []
    title_mark = 0
    yield_confidence = 1
    entits_counter = 0
    
    
    for enty, mark, char, char_end in entits:
        entits_counter += 1
        enty = str(enty)
        
        
        if mark in reags[:2] and yield_counter == 0:
            enty = re.sub(r'   ', r'__', enty)
            enty = re.sub(r'  ', r'__', enty)
            reagents.append([re.sub(r' ', r'__', enty), mark])
            
            if mark == 'COMPOUND':
                comps_list.append(re.sub(r' ', r'__', enty))
                
            if len(reagents) == 2:
                first_reagent = [re.sub(r' ', r'__', enty), mark]
                
            if last_mark[0] in ['QUANTITY', 'PERCENT'] and char - last_mark[5] < 12 and last_mark[3] > char - last_mark[5]:
                reagents[-1][0] = reagents[-1][0] + ' ' + re.sub(r' ', r'', last_mark[2])
                
                if last_mark[2] in reagents[-2][0].split(' ')[-1]:
                    reagents[-2][0] = reagents[-2][0][:-len(last_mark[2])]
                    
        elif mark in reags[-2:] and yield_counter == 0:
            
            if last_mark[0] in reags[:3] and mark == 'SOLVENT' and reagents[-1][0] != first_reagent[0] and char - last_mark[5] < 18:
                reagents[-1][0] = reagents[-1][0] + '--' + enty
                
            elif mark == 'SOLVENT' and (last_mark[0] == 'PERCENT' or 'M' in last_mark[2]) and last_mark[6] == 'MISCINCH':
                reagents[-1][0] = reagents[-1][0].split(' ')[0] + '__' + enty + ' ' + ' '.join(reagents[-1][0].split(' ')[1:])
                
            else:
                reagents.append([enty, mark])
                if last_mark[0] in ['QUANTITY', 'PERCENT'] and char - last_mark[5] < 8 and last_mark[3] > char - last_mark[5]:
                    reagents[-1][0] = reagents[-1][0] + ' ' + re.sub(r' ', r'', last_mark[2])
                    
                    if last_mark[2] in reagents[-2][0].split(' ')[-1]:
                        reagents[-2][0] = reagents[-2][0][:-len(last_mark[2])]
                        
                    if reagents[-2][0].split(' ')[0] == 'Not_recognized':
                        reagents.pop(-2)
                        
        elif mark in ['QUANTITY', 'PERCENT'] and yield_counter == 0 and last_mark[0] in ['AGGREGATION', 'COMPOUND', 'CHEMICALS', 'MISCINCH', 'SOLVENT', 'QUANTITY', 'PERCENT'] and char - last_mark[5] < 8:
            reagents[-1][0] = reagents[-1][0] + ' ' + re.sub(r' ', r'', enty)
            
        elif mark in ['QUANTITY', 'PERCENT'] and yield_counter == 0 and (last_mark[0] not in ['YIELD', 'AGGREGATION', 'COMPOUND', 'CHEMICALS', 'MISCINCH', 'SOLVENT', 'QUANTITY', 'PERCENT'] or char - last_mark[5] >= 8):
            
            cde_in = 0
            for idx, x in enumerate(cde_ends):
                if char - x < 10 and char - x > 0:
                    reagents.append([re.sub(r' ', r'__', cde_entits[idx][0]), 'CHEMICALS'])
                    reagents[-1][0] = reagents[-1][0] + ' ' + re.sub(r' ', r'', enty)
                    cde_in = 1
                    break
                    
            if cde_in == 0:
                reagents.append(['Not_recognized', 'CHEMICALS'])
                reagents[-1][0] = reagents[-1][0] + ' ' + re.sub(r' ', r'', enty)
            
        elif mark in conds and yield_counter <= 0.5:
            conditions.append(enty)
            
            if 'reflux' in enty and reagents[-1][1] == 'SOLVENT':
                conditions[-1] = reagents[-1][0].split(' ')[0] + '__' + enty
            
            if last_mark[0] == 'COMPOUND' and char - last_mark[1] < 8 and  mark == 'TEMPERATURE' and len(reagents) > 2:
                conditions[-1] = last_mark[2] + ' - ' + conditions[-1]
                reagents = reagents[:-1]
                
        elif mark in ['YIELD', 'AGGREGATION'] and (entits_counter > len(entits) * 0.5 or len(entits) < 12):
            
            if mark == 'AGGREGATION':
                yield_counter = 0.5    
            else:
                yield_counter = 1
                
            if mark == 'YIELD' and last_mark[0] in ['QUANTITY', 'PERCENT'] and char - last_mark[5] < 5 and last_mark[3] > char - last_mark[5] and ':' not in last_mark[2] and '/' not in last_mark[2]:
                yields.append(last_mark[2])
                if last_mark[2] in reagents[-1][0].split(' ')[-1]:
                    reagents[-1][0] = reagents[-1][0][:-len(last_mark[2])]
                    
        elif mark == 'PROCEDURE':
            procedures.append(enty)
            
        elif yield_counter >= 0.5 and mark in ['MISCINCH', 'SOLVENT']:
            yield_counter = 0
            reagents.append([enty, mark])
            
        elif yield_counter >= 0.5 and mark in ['COMPOUND', 'CHEMICALS']:
            
            if 'title' not in enty:
                if targ_from_pos in enty or targ_from_pos == '---':
                    yields_dict = {'compound_role': 'target', 'compound_id': enty, 'amounts': []}
                    comp_in_yields_count += 1
                elif targ_from_pos != '---' and targ_from_pos not in enty:
                    yields_dict = {'compound_role': 'target', 'compound_id': enty, 'amounts': []}
                    enty = re.sub(r'   ', r'__', enty)
                    enty = re.sub(r'  ', r'__', enty)
                    reagents.append([re.sub(r' ', r'__', enty), mark])
                    
                
            if 'title' in enty and comp_in_yields_count != 0:
                comps_list.append('')
                yields_dict = {'compound_role': 'target', 'compound_id': comps_list[0] + ' - PTC', 'amounts': []}
                title_mark = 1
                
        elif yield_counter >= 0.5 and mark in ['QUANTITY', 'PERCENT'] and (pattern_1.search(enty) or pattern_2.search(enty)):
            yields.append(enty)
            
#         elif mark == 'AGGREGATION':
#             yields.append(enty)
        else:
            unassigned.append([enty, mark])
            
        last_mark = [mark, char, re.sub(r' ', r'', enty), char - last_mark[5], last_mark[2], char_end, last_mark[0]]
        
    # Если в тексте явно не был указан целевой продукт, то пытаемся достать его из заголовка
    comps_list.append('')    
    if comp_in_yields_count == 0:
        if first_reagent[1] == 'COMPOUND' and reagents[1][0] == first_reagent[0] and entits[0][1] == 'COMPOUND':
            yields_dict = {'compound_role': 'target', 'compound_id': comps_list[0] + ' - PTC', 'amounts': []} # PTC - probably title compound
        elif first_reagent[1] != 'COMPOUND' or (first_reagent[1] == 'COMPOUND' and (reagents[1][0] != first_reagent[0] or entits[0][1] != 'COMPOUND')):
            yield_confidence = 0
            if len(comps_list) != 1:
                yields_dict = {'compound_role': 'target', 'compound_id': comps_list[0] + ' - JFCT', 'amounts': []} # JFCT - just first compound in text
            else:
                yields_dict = {'compound_role': 'target', 'compound_id': comps_list[0] + ' - Target not stated', 'amounts': []} # В методике либо не указан вообще какой-то конкретный компаунд (мб есть отсылка
                                                                                                                            # на общую методику), либо модель его не выявила
                
    # Удаляем из списка с реагентами те компаунды, которые можно отнести к целевым продуктам
    # Удаляем часть информации, относящейся к очистке
    R = copy.deepcopy(reagents)
    n = 0
    for reagent in reagents:
        if (reagent[0].split(' ')[0] == yields_dict['compound_id'] and (targ_from_pos == '---' or targ_from_pos == yields_dict['compound_id'])): #or targ_from_pos in reagent[0].split(' ')[0]:
            R.pop(n)
            n -= 1
        elif '/' in reagent[0].split(' ')[0] or ':' in reagent[0].split(' ')[0]:
            unassigned.append(reagent)
            R.pop(n)
            n -= 1
        elif comp_in_yields_count == 0 and yields_dict['compound_id'].split(' - ')[0] == reagent[0].split(' ')[0] and yield_confidence == 1:
            if len(reagent[0].split(' ')) > 1:         # Позволяет не терять количественную информацию о выходе
                yields_add = reagent[0].split(' ')[1:]
                yields = yields_add + yields
            if targ_from_pos == '---' or targ_from_pos == yields_dict['compound_id'].split(' - ')[0]:
                R.pop(n)
            n -= 1
        n += 1
    reagents = R
    
    reagents_dict = []
    for entity in reagents[1:]:
        entity[0] = entity[0].split(' ')
        reagent_dict = {'compound_role': entity[1], 'compound_id': entity[0][0], 'amounts': entity[0][1:]}
        reagents_dict.append(reagent_dict)
    
    # Этот блок убирает из 'amounts' те значения, которые не должны были туда попасть
    D = copy.deepcopy(reagents_dict)
    for r in range(len(D)):
        t = 0
        for amount in reagents_dict[r]['amounts']:
            if bool(pattern_1.search(amount)) == False and bool(pattern_2.search(amount)) == False:
                unassigned.append([amount, 'AMOUNT'])
                D[r]['amounts'].pop(t)
                t -= 1
            t += 1
    reagents_dict = D
    
    unassigned_dict = []
    for enty, mark in unassigned:
        unassigned_dict.append({'role': mark, 'id': enty})
       
    yields_dict['amounts'] = yields
    yields_dict['posed_compound_id'] = targ_from_pos 
            
    # Решает кейс, когда таргет вначале и к нему там же привязаны количества
    try:
        if ('PTC' in yields_dict['compound_id'] or yields_dict['posed_compound_id'] != '---') and (reagents_dict[0]['compound_id'] in yields_dict['compound_id'] or reagents_dict[0]['compound_id'] in yields_dict['posed_compound_id']) and len(reagents_dict[0]['amounts']) > 0:
            yields_dict['amounts'] = yields_dict['amounts'] + reagents_dict[0]['amounts']
            reagents_dict = reagents_dict[1:]
    except:
        yields_dict['amounts'] = yields
        
    
    return {'REAGENTS': reagents_dict, 'YIELDS': yields_dict, 'PROCEDURE NAME': procedures, 'CONDITIONS': conditions, 'OTHER': unassigned_dict}


def assignment(protocols_list): # ссылка на файл json с методиками

    last_general = []
    gen_names = []
    gen_names_idx = []
    protocol_dicts = []
    
    for idx in range(len(protocols_list)):
        
        dist = protocols_list[idx]['dist_before']
        protocol = protocols_list[idx]['protocol']
        protocol_dict = ner_to_dict(protocol)
#         print(protocol_dict)
#         print(protocol)
        protocol = re.sub(r'  ', r' ', protocol)
        words = protocol.split(' ')
        protocol_dict['id'] = idx
        protocol_dict['refers_to'] = -1
        '''
        i = -1 - методика является общей или не имеет отсылок на другие
        i = 0...n - было определено, что методика, вероятно, ссылается на общую с индексом i
        '''
        protocol_dict['reference_confidence'] = 'not_determined'
        ''' 
        not_determined - методика самостоятельна, или не имеет признаков принадлежности к общей, или сама является общей
        strong - найдена прямая отсылка на четко обозначенную общую методику, вероятность ошибочной ссылки довольно низка
        weak - для установления взаимосвязи были использованы rule-based методы разной степени уверенности, вероятность ошибочной ссылки высокая
        '''
        protocol_dict['is_general'] = False
        guess_proc = [n for n in range(len(words)) if 'rocedure' in words[n] or 'reparation' in words[n]]
        guess_synt = [n for n in range(len(words)) if 'ynthesis' in words[n] and 'of' in words[n+1]]
        guess_prev = [n for n in range(len(words)) if 'rocedure' in words[n] and 'above' in words[n-1]]
        guess_meth = [n for n in range(len(words)) if 'ethod' in words[n] and '(' in words[n+1] and (len(words[n+1]) == 3 or len(words[n+1]) == 4)]
        synt_count = 0
        for i, name in enumerate(protocol_dict['PROCEDURE NAME']):
            if 'ynthesis of' in name:
                protocol_dict['PROCEDURE NAME'][i] = ' '.join(words[guess_synt[synt_count] : guess_synt[synt_count] + 3])
                synt_count += 1
        
        if len(protocol_dict['REAGENTS']) == 0 and len(protocol_dict['YIELDS']['amounts']) == 0:
            protocol_dicts.append(protocol_dict)
            continue
    
        elif protocol_dict['PROCEDURE NAME'] != [] and (len(protocol_dict['YIELDS']['amounts']) == 0 or '~' in protocol_dict['YIELDS']['amounts'][-1] or '-' in protocol_dict['YIELDS']['amounts'][-1]):
            protocol_dict['is_general'] = True
            
            if len(protocol_dict['PROCEDURE NAME']) == 1 and 'for' not in protocol_dict['PROCEDURE NAME'][0]:
                gen_names.append(protocol_dict['PROCEDURE NAME'][0].lower())
                gen_names_idx.append((protocol_dict['PROCEDURE NAME'][0].lower(), idx))
                
            elif len(protocol_dict['PROCEDURE NAME']) == 1 and 'for' in protocol_dict['PROCEDURE NAME'][0]:
                if 'the' in words[guess_proc[0] + 2].lower():
                    protocol_dict['PROCEDURE NAME'] = ' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 6])
                    gen_names.append(protocol_dict['PROCEDURE NAME'].lower())
                    gen_names_idx.append((protocol_dict['PROCEDURE NAME'].lower(), idx))
                else:
                    protocol_dict['PROCEDURE NAME'] = ' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 5])
                    gen_names.append(protocol_dict['PROCEDURE NAME'].lower())
                    gen_names_idx.append((protocol_dict['PROCEDURE NAME'].lower(), idx))
                
            elif len(protocol_dict['PROCEDURE NAME']) > 1:
                for i, proc_name in enumerate(protocol_dict['PROCEDURE NAME']):
                    if 'for' not in proc_name:
                        protocol_dict['PROCEDURE NAME'][i] = proc_name    
                    elif 'for' in proc_name and 'the' in words[guess_proc[0] + 2].lower():
                        proc_name = ' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 6])
                        protocol_dict['PROCEDURE NAME'][i] = proc_name
                    else:
                        proc_name = ' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 5])
                        protocol_dict['PROCEDURE NAME'][i] = proc_name
                    gen_names.append(proc_name.lower())
                    gen_names_idx.append((proc_name.lower(), idx))
                    
            last_general = (protocol_dict['PROCEDURE NAME'], idx)
            
        elif protocol_dict['PROCEDURE NAME'] == [] and len(guess_prev) > 0 and len(protocol_dict['YIELDS']['amounts']) > 0 and len(last_general) > 1:
            protocol_dict['PROCEDURE NAME'] = last_general[0]
            protocol_dict['refers_to'] = last_general[1]
            protocol_dict['reference_confidence'] = 'weak'

        elif protocol_dict['PROCEDURE NAME'] == [] and len(guess_proc) > 0:
            if len(protocol_dict['YIELDS']['amounts']) > 0 and '~' not in protocol_dict['YIELDS']['amounts'][-1] and '-' not in protocol_dict['YIELDS']['amounts'][-1]:
                protocol_dict['PROCEDURE NAME'] = ' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 5])
                for gen_name in gen_names:
                    if gen_name in protocol_dict['PROCEDURE NAME'].lower():
                        protocol_dict['PROCEDURE NAME'] = gen_name
                        protocol_dict['refers_to'] = gen_names_idx[gen_names.index(gen_name)][1]
                        protocol_dict['reference_confidence'] = 'strong'
                        break
                
            if len(protocol_dict['YIELDS']['amounts']) == 0 or '~' in protocol_dict['YIELDS']['amounts'][-1] or '-' in protocol_dict['YIELDS']['amounts'][-1]:
                protocol_dict['PROCEDURE NAME'] = ' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 5])
                protocol_dict['is_general'] = True
                
                if 'tarting material' in protocol.lower() or 'taring material' in protocol.lower():
                    last_general = (' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 5]), idx)
                    gen_names.append(' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 5]))
                    gen_names_idx.append((' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 5]), idx))
                    
        elif protocol_dict['PROCEDURE NAME'] == [] and len(guess_meth) > 0:
            protocol_dict['PROCEDURE NAME'] = ' '.join(words[guess_meth[0] : guess_meth[0] + 2])
            for gen_name in gen_names:
                    if protocol_dict['PROCEDURE NAME'].lower()[-4:-1] in gen_name:
                        protocol_dict['PROCEDURE NAME'] = gen_name
                        protocol_dict['refers_to'] = gen_names_idx[gen_names.index(gen_name)][1]
                        protocol_dict['reference_confidence'] = 'strong'
                        break
            
            
        elif protocol_dict['PROCEDURE NAME'] != [] and len(protocol_dict['YIELDS']['amounts']) > 0:
            if len(protocol_dict['PROCEDURE NAME']) == 1 and protocol_dict['PROCEDURE NAME'][0].lower() in gen_names:
                protocol_dict['refers_to'] = gen_names_idx[gen_names.index(protocol_dict['PROCEDURE NAME'][0].lower())][1]
                protocol_dict['reference_confidence'] = 'strong'
            
            elif protocol_dict['PROCEDURE NAME'][0].lower() not in gen_names or len(protocol_dict['PROCEDURE NAME']) == 2:
                for gen_name in gen_names:
                    if len(protocol_dict['PROCEDURE NAME']) == 1 and gen_name in protocol_dict['PROCEDURE NAME'][0].lower():
                        protocol_dict['PROCEDURE NAME'] = gen_name
                        protocol_dict['refers_to'] = gen_names_idx[gen_names.index(gen_name)][1]
                        protocol_dict['reference_confidence'] = 'strong'
                        break
                    elif len(protocol_dict['PROCEDURE NAME']) == 1 and len(guess_proc) > 0 and ' '.join(words[guess_proc[0] : guess_proc[0] + 2]).lower() in gen_name and len(words[guess_proc[0] + 1]) <= 2:
                        protocol_dict['PROCEDURE NAME'] = gen_name
                        protocol_dict['refers_to'] = gen_names_idx[gen_names.index(gen_name)][1]
                        protocol_dict['reference_confidence'] = 'strong'
                        break
                    elif len(protocol_dict['PROCEDURE NAME']) == 2 and (gen_name in protocol_dict['PROCEDURE NAME'][0].lower() or gen_name in protocol_dict['PROCEDURE NAME'][1].lower()):
                        protocol_dict['PROCEDURE NAME'] = gen_name
                        protocol_dict['refers_to'] = gen_names_idx[gen_names.index(gen_name)][1]
                        protocol_dict['reference_confidence'] = 'strong'
                        break
            
        elif protocol_dict['PROCEDURE NAME'] == [] and len(protocol_dict['YIELDS']['amounts']) > 0:
            if len(guess_proc) == 0 and len(guess_synt) == 0 and len(gen_names) >= 1:
                protocol_dict['PROCEDURE NAME'] = last_general[0]
                protocol_dict['refers_to'] = last_general[1]
                protocol_dict['reference_confidence'] = 'weak'
            elif len(guess_proc) == 0 and len(guess_synt) != 0:
                protocol_dict['PROCEDURE NAME'] = ' '.join(words[guess_synt[0] : guess_synt[0] + 3])
            elif len(guess_proc) != 0 and len(gen_names) >= 1:
                protocol_dict['PROCEDURE NAME'] = [' '.join(words[guess_proc[0] - 1 : guess_proc[0] + 5]), last_general[0]]
                protocol_dict['refers_to'] = last_general[1]
                protocol_dict['reference_confidence'] = 'weak'
                
        protocol_dicts.append(protocol_dict)
    
    return protocol_dicts
