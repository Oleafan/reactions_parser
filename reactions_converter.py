#внешние либы
import re
from rdkit import Chem 
from rdkit.Chem.Descriptors import MolWt 

import numpy as np
import copy
from itertools import permutations, combinations  

#для контроля реакций
from rxnmapper import RXNMapper 
rxn_mapper = RXNMapper()

from xgboost import XGBClassifier

import pickle
with open('/media/oleg/second_ssd/rxn_validator/xgb200_01.pkl', 'rb') as f:
    xgb = pickle.loads(f.read())
from drfp import DrfpEncoder

from  indigo import Indigo
indigo = Indigo()
indigo.setOption('molfile-saving-mode', 3000)
import json
from itertools import permutations, combinations  

import dictionary as dw

#температуры кипения - надо для расшифровки reflux в conditions
indigo_solv_bp = {}
for smiles in dw.solv_bp:
    indigo_solv_bp[indigo.loadMolecule(smiles)] =  dw.solv_bp[smiles]
    
solv_mols = list(indigo_solv_bp.keys())
for mol in solv_mols:
    mol.aromatize()
    
#сюда относятся вспомогательные вещества типа Na2SO4 - оно явно не должно входить в реакцию    
aux_mols = []
for inch in dw.auxilliary_mols:
    aux_mols.append(indigo.loadMolecule(inch))

for mol in aux_mols:
    mol.aromatize()

waste_mols = []
for inch in dw.solv_inch: #тут не только солвенты, но и всякие другие молекулы, которые явно не компаунды
    waste_mols.append(indigo.loadMolecule(inch))
for mol in waste_mols:
    mol.aromatize()

def _transform_reaction_dict(protocol):
    transformed_protocol_list = []
    try:
        parsed_reactions = json.loads(protocol['parsed'])['reactions']
    except Exception as e:
        return []
    for idx, reaction in enumerate(parsed_reactions):
        temp_reaction = {}
        temp_reaction['procedure'] = protocol['protocol']
        if idx == 0:
            temp_reaction['dist_before'] = protocol['dist_before']
        else:
            temp_reaction['dist_before'] = 0
        if 'REAGENTS' not in reaction or 'YIELDS' not in reaction:
            continue
        temp_reaction['compounds'] = reaction['REAGENTS'] + reaction['YIELDS']
        temp_reaction['compounds'] = [x for x in temp_reaction['compounds'] if ('compound_id' in x and 'compound_role' in x)]
        if len(temp_reaction['compounds']) > 30:
            temp_reaction['compounds'] = []        
        
        if 'PROCEDURE NAME' in reaction:
            temp_reaction['procedure_name'] = [x for x in reaction['PROCEDURE NAME'] if (not re.search(r'[Ss]ynthesis\sof', x))]
        else:
            temp_reaction['procedure_name'] = []
        temp_reaction['conditions'] = reaction['CONDITIONS']
        temp_reaction['idx'] = protocol['idx']
        transformed_protocol_list.append(temp_reaction)
    return transformed_protocol_list

def _replace_gen_titles(text):
    sent_text = text.lower()
    sent_text = re.sub(r'\s+', ' ', sent_text)
    #DEFENITELY NOT GP:
    sent_text = re.sub(r'the\s+general\s', 'the_general_', sent_text)
    sent_text = re.sub(r'using\s+general\s', 'using_general_', sent_text)
    sent_text = re.sub(r'following\s+general\s', 'following_general_', sent_text)
    sent_text = re.sub(r'to\s+general\s', 'to_general_', sent_text)
    sent_text = re.sub(r'from\s+general\s', 'from_general_', sent_text)
    sent_text = re.sub(r'general\s+procedure.{1,8}was\s', 'general_procedure_was_', sent_text)
    sent_text = re.sub(r'typical\s+procedure.{1,8}was\s', 'typical_procedure_was_', sent_text)
    
    sent_text = re.sub(r'by\s+general\s', 'by_general_', sent_text)

    sent_text = re.sub(r'the\s+typical\s', 'the_typical_', sent_text)
    sent_text = re.sub(r'using\s+typical\s', 'using_typical_', sent_text)
    sent_text = re.sub(r'following\s+typical\s', 'following_typical_', sent_text)
    sent_text = re.sub(r'to\s+typical\s', 'to_typical_', sent_text)
    sent_text = re.sub(r'from\s+typical\s', 'from_typical_', sent_text)   
    sent_text = re.sub(r'by\s+typical\s', 'by_typical_', sent_text)
    return sent_text

def _is_general(protocol, proc_names):        
    sent_text = _replace_gen_titles(protocol)
    specials = ['.', ',', '-', '[', ']', '(', ')', ';', '*', '%', ':']
    
    for proc_name in proc_names: 
        new_proc_name = proc_name.lower()
        for special in specials:
            new_proc_name =  new_proc_name.replace(special, '\\' + special)
        try:
            if re.search(r'according\sto(?:\sthe)?\s' + new_proc_name, sent_text):
                return False
            if re.search(r'following(?:\sthe)?\s' + new_proc_name, sent_text):
                return False
            if re.search(r'in accordance\swith(?:\sthe)?\s' + new_proc_name, sent_text):
                return False    
        except Exception as e:
            print(e)
            print(proc_name.lower())
    if re.search(r'\bgeneral\s+procedure', sent_text):
        return True
    if re.search(r'\bgeneral\s+protocol', sent_text):
        return True
    if re.search(r'\btypical\s+procedure', sent_text):
        return True
    if re.search(r'\btypical\s+protocol', sent_text):
        return True
    for proc_name in proc_names:
        if '(' + proc_name + ')' in protocol:
            return True
        
    return False


def _parse_conditions(string, compid_descr): #compid_descr - sorted by lenght of key, ascending = False!!!
    #на вход идет строка вида '24mmol'. На выход - словарь вида {'amount': 24 }
    #стандартные единицы измерения количеств: mass - mg, volume - mL, amount - mmol, equiv - в %
    
    for item in ['room temper', 'r.t', 'r. t', 'ambient']:
        if item in string:
            return {'temperature': 25}
    #проверка на reflux
    #'Solvent_4__under reflux'
    if 'Solvent_' in string and 'refl' in string:
        solv_regex = r'Solvent_\d+'
        solv_code = re.findall(solv_regex, string)[0][:-1]
        if solv_code in compid_descr:
            solv_smi = compid_descr[solv_code]  
            solv_mol = indigo.loadMolecule(solv_smi)
            solv_mol.aromatize()
            for mol in indigo_solv_bp:
                mol.aromatize()
                if indigo.exactMatch(solv_mol, mol):
                    return {'temperature': indigo_solv_bp[mol]} 
    
    string = _name_to_num(string)
    
    re_digit = r'\-?\s*\d+(?:\.\d+)?' #-4.5
    digits = re.findall(re_digit, string)    
    if len(digits) == 1:
        try:
            result = float(digits[0])
        except:
            return {'unproc_cond': string}    
        rest_string = re.sub(re_digit, ' ', string)
        # темпеартуры - в °С
        for item in ['°', 'oC', '°C', '°C.', '°C.', 'ºC', '℃', '°C']:
            if item in rest_string:
                return {'temperature': result}
        #давления - в бар
        if 'mbar' in rest_string:
            return {'pressure': result/1000}    
        for item in ['bar', 'atm']:
            if item in rest_string:
                return {'pressure': result}            
        #время - в часы
          
        for item in ['hour', 'hours', ' h']:
            if item in rest_string:
                return {'time': result}    
        for item in ['min', 'minute', 'minutes' ]:
            if item in rest_string:
                return {'time': result/60}    
        for item in ['day', 'days']:
            if item in rest_string:
                return {'time': result*24}   
    return {'unproc_cond': string}      

def _name_to_num(string):
    #вспомогательаня функция для превращения буквенных значений в числовые для адекватного парсинга условий.
    str_nums = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}
    for key in str_nums:
        if key + ' ' in string:
            string = string.replace(key, str_nums[key])
            
    return string


#найти все компаунды в compound_id
def _get_smiles(string, compid_descr, code2compid):
    string = string.replace('__', ' ')
    #принимает строку, возвращает список смайлсов, которые в этой строке упомянуты
    specials = ['.', ',', '-', '[', ']', '(', ')', ';', '*', '%', '+']
    border_regex = r'[\s\.\,\;\:\/\(\)=\[\]]'
    string = ' ' + string + ' '
    
    smiles_list = []
    for compid in compid_descr:
        if compid in string:
            if _get_mol_wt(compid_descr[compid]) is not None: #проверка, что молекула адекватная
                molf = indigo.loadMolecule(compid_descr[compid]).molfile()
                if 'RGROUP' not in molf:
                    smiles_list.append(compid_descr[compid])
                    string = string.replace(compid, ' ')

    for code in code2compid:
        if '|' in code or '%' in code or '\\' in code or '+' in code or 'NMR' in code or 'COSY' in code or 'HSQC' in code:
            continue
        if code in ['Me', 'Et', 'Pr', 'Ph', 'Tol']:
            continue
        if code in dw.elements:
            continue
            
        new_code = code
        for special in specials:
            new_code =  new_code.replace(special, '\\' + special)
        
        regex_to_find = border_regex +new_code+ border_regex
        try:        
            if re.search(regex_to_find,string):
                if _get_mol_wt(compid_descr[code2compid[code]]) is not None:
                    molf = indigo.loadMolecule(compid_descr[code2compid[code]]).molfile()
                    if 'RGROUP' not in molf:
                        smiles_list.append(compid_descr[code2compid[code]])
                        string = re.sub(regex_to_find, ' ', string)
        except Exception as e:
            print(e, '\n', regex_to_find, new_code)
            
    return list(set(smiles_list))

def _get_mol_wt(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    if mol is None:
        try:
            mol = Chem.MolFromInchi(smiles)
        except:
            return None
    if mol is None: 
        return None    
    molwt = round(MolWt(mol), 1)
    return molwt 

def _parse_amounts(string, type_comp = 'COMPOUND'):
    #на вход идет строка вида '24mmol'. На выход - словарь вида {'amount': 24 }
    #стандартные единицы измерения количеств: mass - mg, volume - mL, amount - mmol, equiv - в %

    re_digit = r'\d+(?:\.\d+)?'
    digits = re.findall(re_digit, string)
    if len(digits) == 1:
        try:
            result = float(digits[0])
        except:
            return None
        
        rest_string = re.sub(re_digit, ' ', string)
        #понять что это
        
        #equiv - to mol%
        if ' mmol%' in rest_string:
            return {'mol%': result/1000}
        elif ' mol%' in rest_string:
            return {'mol%': result}
        elif ' mol %' in rest_string:
            return {'mol%': result}
        elif ' equiv' in rest_string or 'eq.' in rest_string:
            return {'mol%': result*100}
        elif 'wt.%' in rest_string:
            return {'mass_conc': result}
        elif ' %' in rest_string and type_comp == 'target':
            return {'mol%': result}
        elif ' %' in rest_string : #for everething except for targets
            return {'mass_conc': result}
        
        #amounts - to mmol
        if ' mmol' in rest_string:
            return {'amount': result}
        elif ' µmol' in rest_string:
            return {'amount': result/1000}
        elif ' mol' in rest_string:
            return {'amount': result*1000}
        
        #mass - to mg
        if ' mg' in rest_string:
            return {'mass': result}
        elif ' µg' in rest_string:
            return {'mass': result/1000}
        elif ' mcg' in rest_string:
            return {'mass': result/1000}
        elif ' g' in rest_string:
            return {'mass': result*1000}
        
        # volume - to mL
        if ' mL' in rest_string:
            return {'volume': result}
        elif ' ml' in rest_string:
            return {'volume': result}
        elif ' µL' in rest_string:
            return {'volume': result/1000}
        elif ' µl' in rest_string:
            return {'volume': result/1000}
        elif ' mcl' in rest_string:
            return {'volume': result/1000}
        elif ' mcL' in rest_string:
            return {'volume': result/1000}
        elif ' L' in rest_string:
            return {'volume': result*1000}        
        
        #Concentration - to M
        if ' M' in rest_string:
            return {'conc': result}
        
        
def _product(input_list):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in input_list] 
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
        if len(result) > 200:
            return [[x[0] for x in input_list]]
    return result


def _parse_reaction_comp_lists (reaction,  compid_descr, code2compid, compid_to_name):
    #функция заменяет обозначения компаундов на смайлсы и оцифровывает количества 
    resulting_reaction = {}
    protocol = reaction['procedure']
    for key in compid_to_name:
        protocol = protocol.replace(key, compid_to_name[key] )
    resulting_reaction['procedure'] = protocol

    conditions_list = []
    for cond in reaction['conditions']:
        if type(cond) == str:
            conditions_list.append(_parse_conditions(cond, compid_descr))
    resulting_reaction['conditions'] = conditions_list
    #вытащить смайлсы и количества + первичная расстановка score

    compounds_smi = []
    for comp in reaction['compounds']:
        try:
            if 'compound_id' not in comp or 'amounts' not in comp or 'compound_role' not in comp:
                continue
            smiles_list = _get_smiles(comp['compound_id'], compid_descr, code2compid)
            digit_amounts = {}
            for amount in comp['amounts']:
                parse_res = _parse_amounts(amount,comp['compound_role'])
                if parse_res:
                    for key in parse_res:
                        if key not in digit_amounts:
                            digit_amounts.update(parse_res)

            score = 0
            if len(smiles_list) == 1 and comp['compound_role'] == 'target':
                score += 2 #only one target in the list

            if len(smiles_list) > 1:  
                #first variant - solvent in the compounds list
                #in this variant volume is attributed to solvent and other amounts - to compound
                #here should be both solvent and nonsolvent

                solvent_found = False
                nonsolvent_found = False
                temp_comp_smi = []

                for smiles in smiles_list:
                    comp_dict = {}
                    comp_dict['score'] = score
                    comp_dict['smiles'] = [smiles]
                    comp_dict['compound_role'] = comp['compound_role'] 
                    comp_dict['digit_amounts'] = copy.deepcopy(digit_amounts)      

                    #check if this is solvent
                    is_solvent = False
                    mol = indigo.loadMolecule(smiles)
                    mol.aromatize()
                    for solv_mol in solv_mols:
                        if indigo.exactMatch(solv_mol, mol):
                            comp_dict['compound_role'] = 'SOLVENT'
                            is_solvent = True
                            solvent_found = True
                            if 'volume' in comp_dict['digit_amounts']:
                                comp_dict['digit_amounts'] = {'volume': comp_dict['digit_amounts']['volume']}
                            else:
                                comp_dict['digit_amounts'] = {}        
                            break

                    if not is_solvent:
                        nonsolvent_found = True
                        comp_dict['digit_amounts'].pop('volume', None)

                    temp_comp_smi.append(comp_dict)
                if solvent_found and nonsolvent_found:
                    compounds_smi += temp_comp_smi
                else:
                    comp_dict = {}
                    comp_dict['score'] = score
                    comp_dict['smiles'] = smiles_list
                    comp_dict['compound_role'] = comp['compound_role'] 
                    comp_dict['digit_amounts'] = copy.deepcopy(digit_amounts)      
                    compounds_smi.append(comp_dict)            

            elif len(smiles_list)> 0:
                #for smiles in smiles_list:
                comp_dict = {}
                comp_dict['score'] = score
                comp_dict['smiles'] = smiles_list
                comp_dict['compound_role'] = comp['compound_role'] 
                comp_dict['digit_amounts'] = copy.deepcopy(digit_amounts)      
                compounds_smi.append(comp_dict)

            elif len(digit_amounts) > 0:
                comp_dict = {}
                comp_dict['score'] = score
                comp_dict['smiles'] = [None]
                comp_dict['compound_role'] = comp['compound_role'] 
                comp_dict['digit_amounts'] = copy.deepcopy(digit_amounts) 
                compounds_smi.append(comp_dict)
        except Exception as e:
            pass
    for comp in compounds_smi:
        if 'mol%' in comp['digit_amounts'] and 'mass' in comp['digit_amounts']:
            if comp['digit_amounts'] == 0 and comp['digit_amounts']['mass'] > 0:
                comp['score'] += -10


    #если у нас в таргетах есть смайлс, который есть в списке в каком-то компаунде - этот список удаляем целиком.
    targets = []
    for comp in compounds_smi:
         if comp['compound_role'] == 'target' and comp['smiles'][0] is not None:
            targets += comp['smiles']

    clean_compounds = []
    for comp in compounds_smi:
        #убрать vol из тех мест, где есть масса

        if 'mass' in comp['digit_amounts'] and 'volume' in comp['digit_amounts']:
            if comp['digit_amounts']['volume'] > comp['digit_amounts']['mass']/2 and \
                comp['digit_amounts']['volume'] < comp['digit_amounts']['mass']*2:
                comp['score'] += 1

            comp['digit_amounts'].pop('volume', None)
        if comp['compound_role'] != 'target' and comp['smiles'][0] is not None:
            for smiles in comp['smiles']:
                if smiles in targets:
                    comp['smiles'] = [None]
                    break
        if 'smiles' in comp and comp['smiles'][0] is not None:
            new_smi_list = []
            for smiles in comp['smiles']:
                if _get_mol_wt(smiles) is not None:
                    new_smi_list.append(smiles) #удалить нечитаемые смайлсы
            if len(new_smi_list) > 0:
                comp['smiles'] = new_smi_list
            else:
                comp['smiles'] = [None]

        if 'smiles' in comp and comp['smiles'][0] is None and len(comp['digit_amounts']) == 0:
            continue

        clean_compounds.append(comp)


    #плодим варианты реакций: если у нас есть один компаунд, в котором оказалось два смайлса - 
    #это повод рассмотреть два варианта реакции. Для перебора используем самодельную функцию _product

    smiles_list_list = [comp['smiles'] for comp in clean_compounds]
#     print(smiles_list_list)
    smiles_combs = _product(smiles_list_list)
    react_cands = []
    for idx_glob, smiles_comb in enumerate(smiles_combs):
        if idx_glob > 10: 
            break # скорее всего это какой-то трэш. Чтобы не зависло делаю такое ограничение
#         try: 
        #ломается когда срабатывает ограничение в функциии _product. ХЗ  пччему так. 
        clean_compounds_local = copy.deepcopy(clean_compounds)
        for idx, comp in enumerate(clean_compounds_local):
            comp['smiles'] = smiles_comb[idx]
        clean_compounds_local = [comp for comp in clean_compounds_local if (comp['smiles'] is not None or len(comp['digit_amounts']) > 0)]
        react_cands.append(clean_compounds_local)
#         except Exception as e:
#             pass

    return resulting_reaction, react_cands

def _initial_hole_fill(clean_compounds):
    #если для компаундов много что и так прописано напрямую - заполнить прямыми соответствиями
    #если для продукта известна масса и количество -  вычислить молярную массу 
    #и попробовать подобрать под нее компаунд
    unassign_smi = [comp['smiles'] for comp in clean_compounds if (comp['compound_role'] != 'SOLVENT' and 
                                                                       len(comp['digit_amounts']) == 0)] 
    for comp in clean_compounds:
        if comp['compound_role'] == 'target' and comp['smiles'] is None:
            if 'mass' in comp['digit_amounts'] and 'amount' in comp['digit_amounts'] and comp['digit_amounts']['amount'] != 0:
                mol_weight_predicted = comp['digit_amounts']['mass']/comp['digit_amounts']['amount']
                for smiles in unassign_smi:
                    molwt = _get_mol_wt(smiles)
                    if molwt is not None and np.abs(molwt-mol_weight_predicted ) < mol_weight_predicted*0.03:
                        comp['smiles'] = smiles
                        break    
    
    #то же самое сделать под компаунды без смайлсов
    unassign_smi = [comp['smiles'] for comp in clean_compounds if (comp['compound_role'] != 'SOLVENT' and 
                                                                   len(comp['digit_amounts']) == 0)] 
    for comp in clean_compounds:
        if (comp['compound_role'] != 'SOLVENT') and comp['smiles'] is None:
            if 'mass' in comp['digit_amounts'] and 'amount' in comp['digit_amounts']:
                if comp['digit_amounts']['amount'] != 0:
                    mol_weight_predicted = comp['digit_amounts']['mass']/comp['digit_amounts']['amount']
                    for smiles in unassign_smi:
                        molwt = _get_mol_wt(smiles)
                        if comp['digit_amounts']['amount'] >= 0.1: 
                            tolerance = 0.03
                        else:
                            tolerance = 0.1 #для малых количеств погрешность взвешивания иная
                        if np.abs(molwt-mol_weight_predicted ) < mol_weight_predicted*tolerance:
                            comp['smiles'] = smiles
                            break
    
    #пройтись по компаундам. И если среди реагентов есть смайлс продукта - убрать нафиг
    assigned_targets = [comp['smiles'] for comp in clean_compounds if (comp['compound_role'] == 'target' and 
                                                                       comp['smiles'] is not None)]
    for comp in clean_compounds:
        if comp['compound_role'] != 'target' and comp['smiles'] in assigned_targets:
            comp['smiles'] = None
    
    #удалить компаунды, которые стоят без digit_amounts при том, что в другом месте они стоят с digit_amounts
    assigned_comps = [comp['smiles'] for comp in clean_compounds if (len(comp['digit_amounts']) > 0 and 
                                                                       comp['smiles'] is not None)]
    for comp in clean_compounds:
        if comp['smiles'] in assigned_comps and len(comp['digit_amounts']) == 0:
            comp['smiles'] = None    
    
    #удалить компаунды для которых smiles и digit_amounts нулевые
    clean_compounds = [comp for comp in clean_compounds if (comp['smiles'] is not None or 
                                                            len(comp['digit_amounts']) > 0)]
    return clean_compounds


#сопоставить количества между собой внутри одного компаунда. 
def _get_score(smiles, digit_amounts):
    #сопоставить mass & amount
    score = 0
    molwt = _get_mol_wt(smiles)
    clean_mass = None
    if 'mass_conc' in digit_amounts and 'mass' in digit_amounts:
        clean_mass = digit_amounts['mass']*digit_amounts['mass_conc']/100
        
    if 'amount' in digit_amounts and 'mass' in digit_amounts:
        calc_amount = round(digit_amounts['mass']/molwt, 4)
        calc_clean_amount = None
        if clean_mass is not None:
            calc_clean_amount = round(clean_mass/molwt, 4)
        
        if np.abs(calc_amount - digit_amounts['amount'] ) < 0.05*digit_amounts['amount']:
            score += 1
        elif (calc_clean_amount is not None and 
              np.abs(calc_clean_amount - digit_amounts['amount'] ) < 0.05*digit_amounts['amount']):
            score += 1        
        elif np.abs(calc_amount - digit_amounts['amount'] ) < 0.2*digit_amounts['amount']:
            score += 0.2
        elif (calc_clean_amount is not None and 
              np.abs(calc_clean_amount - digit_amounts['amount'] ) < 0.2*digit_amounts['amount']):
            score += 0.2
        else:
            score += -1
    
    elif 'volume' in digit_amounts and 'amount' in digit_amounts:
        calc_amount = round(digit_amounts['volume']*1000/molwt, 4)
        if np.abs(calc_amount - digit_amounts['amount'] ) < 0.3*digit_amounts['amount']:
            score += 0.5
        elif np.abs(calc_amount - digit_amounts['amount'] ) < 0.5*digit_amounts['amount']:
            score += 0.2
        else:
            score += -1
    
    if 'volume' in digit_amounts and 'mass' in digit_amounts:
        dens = digit_amounts['mass']/digit_amounts['volume'] 
        if dens < 2 and dens > 0.5:
            score += 0.2
        else:
            score += -1
    return score


def _transform_clean_comps(clean_compounds):
    clean_compounds = copy.deepcopy(clean_compounds)
    #удалить компаунды, которые значатся как таргеты
    targets = [comp['smiles'] for comp in clean_compounds if (comp['compound_role'] == 'target' and 
                                                              comp['smiles'] is not None)]
    clean_compounds = [comp for comp in clean_compounds if (not (comp['compound_role'] != 'target' and 
                                                                 comp['smiles'] in targets)) ]
    #убрать все места, где smiles is None
    clean_compounds = [comp for comp in clean_compounds if comp['smiles'] is not None ] 
    #какие-то шансы найти смайлс для None есть только тогда, когда известна mass, mol% or amount. 
    #Если хоть чего-то из этого нет и comp['smiles'] == None - убрать нафиг. Применимо ко всему кроме таргета
    clean_compounds = [comp for comp in clean_compounds if not (comp['compound_role'] != 'target' and 
                                                                     comp['smiles'] is None and
                                                                     'mass' not in comp['digit_amounts'] and 
                                                                     'mol%' not in comp['digit_amounts'] and
                                                                     'amount' not in comp['digit_amounts'] ) ]
    #раздача score
    for comp in clean_compounds:
        if len(comp['digit_amounts']) > 0 and 'smiles' in comp and comp['smiles'] is not None: 
            comp['score'] += _get_score(comp['smiles'], comp['digit_amounts'])
    
    #сопоставлять mol% и amount
    for comp in clean_compounds:
        if len(comp['digit_amounts']) > 0 and comp['score'] >=0:
            digit_amounts = comp['digit_amounts']
            if 'amount' in digit_amounts and 'mol%' in digit_amounts and digit_amounts['amount'] != 0:
                comp['ratio_amount_equiv'] = round(digit_amounts['mol%']/digit_amounts['amount'])
    ratios = np.array([comp['ratio_amount_equiv'] for comp in clean_compounds if 'ratio_amount_equiv' in comp])
    med_ratio = np.median(ratios)
    if len(ratios) > 3:
        for comp in clean_compounds:
            if 'ratio_amount_equiv' in comp:
                if np.abs(comp['ratio_amount_equiv'] - med_ratio) < 0.1*med_ratio:
                    comp['score'] += 0.5
                else: 
                    comp['score'] += -2
    
    #заполнение amount, mol% и mass если их нет
    #если есть масса, но нет молей
    clean_compounds = copy.deepcopy(clean_compounds)
    for comp in clean_compounds:
        if 'mass' in comp['digit_amounts'] and 'amount' not in comp['digit_amounts']:
            molwt = _get_mol_wt(comp['smiles'])
            if molwt:
                comp['digit_amounts']['amount'] = round(comp['digit_amounts']['mass']/molwt,4)
                
    #находим scale - количество ммоль, нормированное на 100% для вещества с максимальным score
    scale = None #не определен
    for comp in clean_compounds:
        if comp['compound_role'] == 'target' and comp['score'] > 1.9:
            if 'mol%' in  comp['digit_amounts'] and 'amount' in  comp['digit_amounts'] and comp['digit_amounts']['mol%'] != 0:
                scale = round(comp['digit_amounts']['amount']/comp['digit_amounts']['mol%']*100, 4)
    
    #print(clean_compounds)
    if scale is None:
        score_list = [comp['score'] for comp in clean_compounds if ('score' in comp and 
                                                                    comp['compound_role'] != 'target')]
        if len(score_list) > 0:
            max_score = max(score_list)
            if max_score:
                for comp in clean_compounds:
                    if comp['score'] == max_score:
                        #print(comp)
                        if 'mol%' in  comp['digit_amounts'] and 'amount' in  comp['digit_amounts']  and comp['digit_amounts']['mol%'] != 0:
                            scale = round(comp['digit_amounts']['amount']/comp['digit_amounts']['mol%']*100, 4)
                        elif 'amount' in  comp['digit_amounts']:
                            scale = round(comp['digit_amounts']['amount'], 4)
                        break
    if scale is None:
        scale_cand = []
        for comp in clean_compounds:
            if ('digit_amounts' in comp and 'amount' in comp['digit_amounts'] and 
                comp['digit_amounts']['amount'] !=0):
                if 'mol%' in comp['digit_amounts'] and comp['digit_amounts']['mol%'] != 0:
                    scale_cand.append(comp['digit_amounts']['amount']/comp['digit_amounts']['mol%']*100)
                else:
                    scale_cand.append(comp['digit_amounts']['amount'])
        if len(scale_cand) == 1:
            scale = scale_cand[0]
        if len(scale_cand) > 1:
            scale_score_dict = {}
            for scale_ in scale_cand:
                scale_score_dict.update({scale_:0})
            for comp in clean_compounds:
                if ('digit_amounts' in comp and 'mol%' in comp['digit_amounts'] and 
                    'amount' in comp['digit_amounts']):
                    for scale_ in scale_score_dict:
                        mol_perc_calc =  comp['digit_amounts']['amount']/scale_*100
                        if np.abs(mol_perc_calc-comp['digit_amounts']['mol%'])/mol_perc_calc < 0.05:
                            scale_score_dict[scale_] += 1
            scale_cand = max(scale_score_dict, key=scale_score_dict.get)
            if scale_score_dict[scale_cand] > 0:
                scale = scale_cand
                
    if scale is None:
        #посмотреть есть ли таргет 
        for comp in clean_compounds:
            if comp['compound_role'] == 'target' and 'amount' in comp['digit_amounts']:
                if 'mol%' in  comp['digit_amounts'] and comp['digit_amounts']['mol%'] != 0:
                    scale = round(comp['digit_amounts']['amount']/comp['digit_amounts']['mol%']*100, 4)
                else:
                    scale = round(comp['digit_amounts']['amount'], 4)
                
    #расчет оставшихся значений
    if scale == 0:
        scale = None
    if scale: 
        for comp in clean_compounds:
            #если есть моли, но нет mol%.
            if 'amount' in comp['digit_amounts'] and 'mol%' not in comp['digit_amounts']:
                comp['digit_amounts']['mol%'] = round(comp['digit_amounts']['amount']/scale*100,1)
            #если есть mol%, но нет молей
            if 'mol%' in comp['digit_amounts'] and 'amount' not in comp['digit_amounts']:
                comp['digit_amounts']['amount'] = round(comp['digit_amounts']['mol%']*scale/100, 4)
            #если есть количество, но нет массы:
            if 'amount' in comp['digit_amounts'] and 'mass' not in comp['digit_amounts']:
                molwt = _get_mol_wt(comp['smiles'])
                comp['digit_amounts']['mass'] = round(comp['digit_amounts']['amount']*molwt,4)   
    
    #сейчас сложилась такая ситуация: есть компаунды с количествами, есть компаунды без количеств. 
    #Повысим score всем компаундам с количествами
    for comp in clean_compounds:
        if 'amount' in comp['digit_amounts']:
            comp['score'] += 0.2   
    
    #сравнить mol% рассчитанные и реальные.
    if scale:
        for comp in clean_compounds:
            if 'mol%' in comp['digit_amounts'] and 'amount' in comp['digit_amounts']:
                ratio_calc = comp['digit_amounts']['amount']/scale*100
                #print(comp['smiles'])
                #print(ratio_calc, comp['digit_amounts']['mol%'])
                if np.abs(ratio_calc - comp['digit_amounts']['mol%'])/ratio_calc > 0.1 and comp['digit_amounts']['mol%'] > 30:
                    comp['score'] += -3
                elif np.abs(ratio_calc - comp['digit_amounts']['mol%'])/ratio_calc > 0.2 and comp['digit_amounts']['mol%'] <= 30:
                    comp['score'] += -3
                elif np.abs(ratio_calc - comp['digit_amounts']['mol%'])/ratio_calc > 0.05 and comp['digit_amounts']['mol%'] > 30:
                    comp['score'] += -0.5
                elif np.abs(ratio_calc - comp['digit_amounts']['mol%'])/ratio_calc > 0.1 and comp['digit_amounts']['mol%'] <= 30:
                    comp['score'] += -0.5
                else: 
                    comp['score'] += 0.35
                    
    #если продукт совпадает с одной  из молекул в waste_mols - понизить его score
    for comp in clean_compounds:
        if comp['compound_role'] == 'target' and 'smiles' in comp:
            #если нет mol% - понизить скор
            if 'mol%' not in comp['digit_amounts']:
                comp['score'] += -2 
            mol = indigo.loadMolecule(comp['smiles']) 
            mol.aromatize()
            for waste_mol in waste_mols:
                if indigo.exactMatch(mol, waste_mol):
                    comp['score'] += -5
                    break
                    
    #посмотреть эквиваленты. Разделить 
    #1) реактанты (все компаунды в количествах 70 моль% и более), 
    #2) реагенты (все мискинч в в количествах 70 моль% и более), 
    #3) катализаторы (все компаунды и мискинчи в количестве меньше 70% ), 
    #4) растворители
    #5) продукты
    for comp in clean_compounds:
        if comp['compound_role'] == 'SOLVENT':
            solv_found = False
            mol = indigo.loadMolecule( comp['smiles'])
            mol.aromatize()
            for solvmol in solv_mols: #duplicated check...
                if indigo.exactMatch(mol, solvmol):
                    solv_found = True
                    break
            if not solv_found:
                comp['compound_role'] = 'CHEMICALS'
        if comp['compound_role'] == 'AUX':
            pass        
        elif comp['compound_role'] == 'MISCINCH':
            if 'mol%' in comp['digit_amounts'] :
                if comp['digit_amounts']['mol%'] >= 70:
                    comp['compound_role'] = 'REAGENT'
                else:
                    comp['compound_role'] = 'CATALYST'
            else:
                comp['compound_role'] = 'UNKNOWN'
        elif comp['compound_role'] == 'COMPOUND' or comp['compound_role'] == 'CHEMICALS':
            if 'mol%' in comp['digit_amounts']:
                if comp['digit_amounts']['mol%'] >= 70:
                    comp['compound_role'] = 'REACTANT'
                else:
                    comp['compound_role'] = 'CATALYST'
            else:
                comp['compound_role'] = 'UNKNOWN'    

        elif comp['compound_role'] == 'target':
            comp['compound_role'] = 'PRODUCT'    
    return clean_compounds, scale


def _get_reaction_score(clean_compounds, limit_score = -3):
    #расчет score для реакции в целом. 
    #1) Сумма всех reagent score
    #2) Если есть продукт - score увеличивается
    #3) подсчет масс атомов. Масса данного элемента слева и справа от стрелки 
    
    score_compounds = 0
    product_exists = 0
    sm_atom_masses = {}
    tm_atom_masses = {}
    for comp in clean_compounds:
        if comp['compound_role'] == 'AUX' or comp['compound_role'] == 'SOLVENT':
            continue
        if 'score' in comp and comp['compound_role'] != 'PRODUCT':
            score_compounds += comp['score']
        if comp['compound_role'] == 'PRODUCT' and len(comp['digit_amounts']) > 0:
            if 'mol%' in comp['digit_amounts'] and comp['digit_amounts']['mol%'] <= 100:
                product_exists = comp['score']
            if 'mol%' in comp['digit_amounts'] and comp['digit_amounts']['mol%'] > 100:
                product_exists = -10 

        if comp['score'] >= limit_score:
            if 'digit_amounts' in comp and 'mass' in comp['digit_amounts']:
                #среди реагентов учитываем только то, для чего mol% 70 и выше. При этом делим на mol% (при  наличии)
                if 'mol%' in comp['digit_amounts'] and comp['digit_amounts']['mol%'] != 0:
                    mass_comp = comp['digit_amounts']['mass']/comp['digit_amounts']['mol%']*100
                else:
                    mass_comp = comp['digit_amounts']['mass']
                composition = _get_atom_masses(comp['smiles'], mass_comp)
                if (comp['compound_role'] != 'PRODUCT' and 'mol%' in comp['digit_amounts'] and 
                    comp['digit_amounts']['mol%'] > 70):
                    for key in composition:
                        if key in sm_atom_masses:
                            sm_atom_masses[key] += composition[key]
                        else:
                            sm_atom_masses[key] = composition[key]
                if comp['compound_role'] == 'PRODUCT':
                    for key in composition:
                        if key in tm_atom_masses:
                            tm_atom_masses[key] += composition[key]
                        else:
                            tm_atom_masses[key] = composition[key]
           
    #сравнивать массы атомов в продуктах и исходниках.
    #насколько сильно от 1:1 отличается с поправкой на CHNO для которых границы сильнее. 
    element_comparison_score = 0 
    #идеальный случай - это полная стехиометрия, и соотношение = 1 для всех элементов. 
    #Метрика по каждому элементу: 
    # (1/(1-х)), где х - соотношение, если  0.5 <= x < 1 . Оно в идеальном случае стремится к беск
    # -1/x если x < 0.5 (тут надо штрафовать модель) (если атом входит в  ['C', 'H', 'N', 'O'] - делить это на 3)
    # -x если x > 1 (тоже надо штрафовать модель) (если атом входит в  ['C', 'H', 'N', 'O'] - делить это на 3)

    elem_score_dict = {}
    score_list = []
    if len(sm_atom_masses) > 0 and len(tm_atom_masses) > 0:
        #перечисляем атомы и считаем соотношение продукт/исходник. 
        elements = list(set(tm_atom_masses.keys()) & set(sm_atom_masses.keys()))
        #print(sm_atom_masses)
        #print(tm_atom_masses)
        for elem in elements:
            if elem in sm_atom_masses and elem in tm_atom_masses and sm_atom_masses[elem] != 0:
                ratio = tm_atom_masses[elem]/1.02/sm_atom_masses[elem] #1.02 позволяет купироватть погрешности взвешивания. Если выход 99% - может случиться, что продукта в 3 знаке больше, чем исходника. 
            elif elem in tm_atom_masses:
                ratio = 100
            else:
                ratio = 0
            score = _atom_score(ratio)
#             if score > 0:
#                 elem_score_dict[elem] = score
#             else:
            if elem in ['C', 'N', 'O']:
                elem_score_dict[elem] = score/3
            elif elem == 'H' and score > 0:
                elem_score_dict[elem] = score/1000
            else:
                elem_score_dict[elem] = score

            score_list.append(elem_score_dict[elem])
    if len(score_list) > 0:
        element_comparison_score = np.median(score_list)
        
        for elem in ['C', 'N', 'H']: 
            if elem in tm_atom_masses and elem not in elem_score_dict:
                element_comparison_score = element_comparison_score/3 #если этот элемент есть в продуктах, но нет в исходниках - это лажа.
        
        element_comparison_score = element_comparison_score*(len(elem_score_dict)/len(set(tm_atom_masses.keys()) | set(sm_atom_masses.keys())))
            
    if product_exists > 0 and len(sm_atom_masses) == 0:
        product_exists = 0
    
    if len( elem_score_dict) == 0 and (len(sm_atom_masses) > 0 or len(tm_atom_masses) > 0):
        element_comparison_score += -10
    if 'C' not in elem_score_dict and ('C' in sm_atom_masses or 'C' in tm_atom_masses):
        element_comparison_score += -10
    
    #где-то тут впиндюрить хемоинформатику для проверки 
    
    try:
        score_chemo = 0
        rxn = _create_reaction(clean_compounds, limit_score)
#         print(rxn.smiles())
        results = rxn_mapper.get_attention_guided_atom_maps([rxn.smiles()])
        if results[0]['confidence'] == 1:
            score_chemo = -5
        else:
            score_chemo =  (results[0]['confidence']-0.4)*5 #it used to be *10 instead of *5
            
        xgb_cl_res = _is_true_rxn(rxn.smiles()) - 0.5 #[-0.5, 0.5]
        score_chemo += xgb_cl_res*8
    except:
        score_chemo  = -5
        xgb_cl_res = 0
    
    
    reagent_count = 0
    product_count = 0
    for comp in clean_compounds:
        if comp['compound_role'] in ['REAGENT', 'CATALYST', 'UNKNOWN', 'REACTANT']:
            reagent_count += 1
        elif comp['compound_role'] == 'PRODUCT':
            product_count += 1
    
    reagents_exist = 0
    if reagent_count ==  0:
        reagents_exist = -5
    
    score_dict_local = {'score_compounds': score_compounds, 
                        'product_exists': product_exists,
                        'element_score': element_comparison_score,
                        'elem_score_dict': elem_score_dict,
                        'sm_atom_masses': sm_atom_masses,
                        'tm_atom_masses': tm_atom_masses, 
                        'score_chemo': score_chemo,
                        'score_xgb': xgb_cl_res,
                        'total_score': score_compounds/2 + product_exists + element_comparison_score + score_chemo + reagents_exist}
    
    return clean_compounds, score_dict_local

def _get_drfp_fp(rxn_smiles):
    try: 
        return DrfpEncoder.encode(rxn_smiles)[0]
    except Exception as e:
        return None
def _is_true_rxn(smiles):
    fp = _get_drfp_fp(smiles)
    return xgb.predict(np.array([fp]))[0]
    
def _atom_score(x):
    # (1/(1-х)), где х - соотношение, если  0.5 <= x < 1 . Оно в идеальном случае стремится к беск
    # -1/x если x < 0.5 
    # -x если x > 1   
    if x >= 0.3 and x < 1:
        return 1/(1-x)
    if x < 0.3 and x > 0:
        return -1/x
    if x >= 1:
        return -x
    if x == 0:
        return 0

def _get_atom_masses(smiles, mass):
    #на выходе словарь вида {'Rh': 0.125} Ключ - атом, значение - масса в г до 5 знака

    mol = indigo.loadMolecule(smiles)
    mwt = mol.molecularWeight()
    formula = mol.grossFormula()
    elements = formula.split()
    composition = {}
    for element in elements:
        el_num = 1
        if len(re.findall(r'\d+', element)) > 0:
            el_symb = re.sub(r'\d+', '', element)
            el_num = int(re.findall(r'\d+', element)[0])
        else: 
            el_symb = element

        submol = indigo.loadMolecule('[' + el_symb + ']')
        composition[el_symb] = round(submol.molecularWeight()*el_num/mwt*mass,5)
        
    return composition


def _create_reaction(clean_compounds, limit_score = 0, get_compounds = 'react'):
    #создание rxn
    rxn = indigo.createReaction()
    if get_compounds == 'react':
        for comp in clean_compounds:
            if comp['compound_role'] == 'PRODUCT' and comp['score'] >= limit_score:
                rxn.addProduct(indigo.loadMolecule(comp['smiles']))  
            elif comp['score'] >= limit_score and comp['compound_role'] not in ['AUX', 'SOLVENT'] :
                if 'mol%' in comp['digit_amounts'] and comp['digit_amounts']['mol%'] > 70:
                    rxn.addReactant(indigo.loadMolecule(comp['smiles']))
                elif comp['compound_role'] in ['REAGENT', 'REACTANT', 'CATALYST', 'UNKNOWN'] and 'mol%' not in comp['digit_amounts']:
                    rxn.addReactant(indigo.loadMolecule(comp['smiles']))
    if get_compounds == 'all':
        for comp in clean_compounds:
            if comp['compound_role'] == 'PRODUCT' and comp['score'] >= limit_score:
                rxn.addProduct(indigo.loadMolecule(comp['smiles']))  
            elif comp['score'] >= limit_score and comp['compound_role'] != 'AUX':
                if comp['compound_role'] in ['REAGENT', 'REACTANT', 'CATALYST', 'UNKNOWN']:
                    rxn.addReactant(indigo.loadMolecule(comp['smiles']))  
                elif comp['compound_role'] == 'SOLVENT' and 'mol%' in comp['digit_amounts']:
                    rxn.addReactant(indigo.loadMolecule(comp['smiles']))  
    return rxn #потом переделать в молфайлы 3000


def _analyze_reaction_list(react_cands):
    #на вход принимает список списков компаундов reaction_smi['compounds']
    
    react_candidates = []  #сюда будем класть перетасованные компаунды. кандидат в реакцию - это формат clean_compounds
    MAX_COMBS = 100
    
    for clean_compounds in react_cands:
        #разметить растворители 
        for comp in clean_compounds:
            if comp['smiles'] is not None:
                mol = indigo.loadMolecule(comp['smiles'])
                mol.aromatize()
                if comp['compound_role'] != 'SOLVENT' and comp['compound_role'] != 'target':
                    for solvmol in solv_mols:
                        if indigo.exactMatch(mol, solvmol):
                            comp['compound_role'] = 'SOLVENT'
                            break
                    for auxmol in aux_mols:
                        if indigo.exactMatch(mol, auxmol):
                            comp['compound_role'] = 'AUX'
                            break    

                elif comp['compound_role'] == 'target':
                    #exclude situations when we have solvent or aux mol as target
                    for solvmol in solv_mols:
                        if indigo.exactMatch(mol, solvmol):
                            comp['smiles'] = None
                            break
                    for auxmol in aux_mols:
                        if indigo.exactMatch(mol, auxmol):
                            comp['smiles'] = None
                            break    
                elif comp['compound_role'] == 'SOLVENT':
                    solv_found = False
                    for solvmol in solv_mols:
                        if indigo.exactMatch(mol, solvmol):
                            solv_found = True
                    if not solv_found:
                        comp['compound_role'] = 'CHEMICALS'

        #Если для компаунда есть только VOL и он большой - убрать
        for comp in clean_compounds:
            if ((comp['compound_role'] == 'COMPOUND' or 
                comp['compound_role'] == 'CHEMICALS' or 
                comp['compound_role'] == 'target') and 
                len(comp['digit_amounts']) == 1 and 
               'volume' in comp['digit_amounts']):
                if comp['digit_amounts']['volume'] >= 50: #Compound_17  A 100 mL round-bottom flask was charged 
                    comp['digit_amounts'] = {}

        #убрать отсюда записи, в которых нет ни digit_amounts, ни smiles
        clean_compounds = [comp for comp in clean_compounds if (comp['smiles'] is not None or 
                                                                len(comp['digit_amounts']) > 0)]

        react_candidates.append(copy.deepcopy(clean_compounds)) #один из вариантов - исходный список

        #если есть таргет без смайлса и нет таргетов со смайлсами - 
        #предложить его в качестве продукта (в начале методики может стоять)
        assigned_targets = [comp['smiles'] for comp in clean_compounds if (comp['compound_role'] == 'target' and 
                                                                           comp['smiles'] is not None)]

        unassigned_targets = [comp['digit_amounts'] for comp in clean_compounds if (comp['compound_role'] == 'target' and 
                                                                                    comp['smiles'] is None and 
                                                                                   'digit_amounts' in comp)]
        if len(assigned_targets) == 0 and len(unassigned_targets) > 0: #нет таргетов со смайлсами и есть просто таргеты
            #список смайлсов без количеств
            unassign_smi = [comp['smiles'] for comp in clean_compounds if (comp['compound_role'] == 'COMPOUND' and 
                                                                           len(comp['digit_amounts']) == 0)] 

            for smiles in unassign_smi:
                clean_compounds_temp = copy.deepcopy(clean_compounds)
                for comp in clean_compounds_temp:
                    if comp['compound_role'] == 'target' and comp['smiles'] is None:
                        comp['smiles'] = smiles
                    if comp['compound_role'] != 'target' and comp['smiles'] == smiles:
                        comp['smiles'] = None #удалить этот смайлс из компаундов

                clean_compounds_temp = [comp for comp in clean_compounds_temp if (comp['smiles'] is not None or 
                                                                                  len(comp['digit_amounts']) > 0)]
                react_candidates.append(clean_compounds_temp)
    
    #если есть какие-то digit_amounts без smiles - сделать выборку компаундов без digit_amounts 
    #и поставить их комбинаторно туда, где есть digit_amounts но без компаунда. 
    #При этом есть ограничение - MAX_COMBS комбинаций
    #После этого потереть все места где нет смайлсов
    
    react_candidates_new = []
    for clean_compounds in react_candidates:
        react_candidates_new.append(copy.deepcopy(clean_compounds))
    
    for idx, clean_compounds in enumerate(react_candidates):
        if idx > MAX_COMBS:
            break
        unassigned_comps_idx = [idx for idx, comp in enumerate(clean_compounds) if (comp['smiles'] is not None and
                                                                                len(comp['digit_amounts']) == 0 and 
                                                                                   comp['compound_role'] != 'SOLVENT' and
                                                                                   comp['compound_role'] != 'AUX')]
        unassigned_da_idx = [idx for idx, comp in enumerate(clean_compounds) if (comp['smiles'] is None and
                                                                                 len(comp['digit_amounts']) > 0 and 
                                                                                 comp['compound_role'] != 'SOLVENT' and
                                                                                 comp['compound_role'] != 'AUX')]
        if len(unassigned_da_idx) > 0:
            #считаем все перестановки списка smiles в том числе None
            unassigned_comps_idx = [None] + unassigned_comps_idx 
            for comb in permutations(unassigned_comps_idx, len(unassigned_da_idx)):
                clean_compounds_temp = copy.deepcopy(clean_compounds)
                
                for idx, position in enumerate(unassigned_da_idx):
                    if comb[idx] is not None:
                        clean_compounds_temp[position]['smiles'] = clean_compounds_temp[comb[idx]]['smiles'] #поставить в слот смайлс
                        clean_compounds_temp[comb[idx]]['smiles'] = None  #убрать отсюда смайлс
                
                react_candidates_new.append(copy.deepcopy(clean_compounds_temp))
                if len(react_candidates_new) > MAX_COMBS:
                    break
        if len(react_candidates_new) > MAX_COMBS:
            break

    finalized_react_cands = []
    scale_list =  []
    for clean_compounds in react_candidates_new:
        new_c_c, scale = _transform_clean_comps(clean_compounds) #рассчитать все, что можно для данной версии clean_compounds
        finalized_react_cands.append(new_c_c)
        scale_list.append(scale)
    
    #считаем скоры реакций
    reactions_scores_dict = {} # {rxn_no: score_dict_local} - with ditalization of scores
    reaction_scores = {} #{rxn_no: total_score}
    reaction_scores_positive = {} #{rxn_no: total_score} if score_compounds is positive
    for idx, clean_compounds in enumerate(finalized_react_cands): 
        clean_compounds, score_dict_local = _get_reaction_score(clean_compounds)
        reactions_scores_dict[idx] = score_dict_local  
        reaction_scores[idx] = score_dict_local['total_score'] 
        if score_dict_local['score_compounds'] > 0:
            reaction_scores_positive[idx] = score_dict_local['total_score'] 
    
    #выбрать реакцию с максимальным total_score
    if len(reaction_scores_positive) > 0:
        idx_max = max(reaction_scores_positive, key=reaction_scores_positive.get)
    else:
        idx_max = max(reaction_scores, key=reaction_scores.get)
        
    processed_reaction = finalized_react_cands[idx_max]
    scale = scale_list[idx_max]
    
    limit_score = -5
    clean_proc_reaction = []
    for comp in processed_reaction:
        if comp['score'] >= limit_score:
            clean_proc_reaction.append(comp) #брать только участников реакции, про которых ничего плохого сказать нельзя
    
    clean_proc_reaction, score_dict = _get_reaction_score(clean_proc_reaction) 
    
    score = score_dict['total_score']
    rxn = _create_reaction(clean_proc_reaction, limit_score = limit_score, get_compounds = 'all')
    
    return clean_proc_reaction, score, rxn, scale, score_dict


def _is_true_reaction(reaction_smi):
    rxn = reaction_smi['rxn']
#     'SOLVENT', 'AUX', 'REAGENT', 'CATALYST', 'UNKNOWN', 'PRODUCT', 'REACTANT' 
#     'AUX' в БД не попадает, 'UNKNOWN' становится 'REAGENT'
    comp_list = reaction_smi['compounds']
    reagent_count = 0
    product_count = 0
    product_yield = False
    for comp in comp_list:
        if comp['compound_role'] in ['REAGENT', 'CATALYST', 'UNKNOWN', 'REACTANT']:
            reagent_count += 1
        elif comp['compound_role'] == 'PRODUCT':
            product_count += 1
            if 'digit_amounts' in comp and 'mol%' in comp['digit_amounts']:
                if comp['digit_amounts']['mol%'] > 0:
                    product_yield = True
            
    if (reaction_smi['score'] > 0 and
        rxn.countReactants() > 0 and 
        rxn.countProducts() > 0 and 
        rxn.countReactants() < 7 and 
        product_count > 0 and
        reagent_count > 0):
        score_dict = reaction_smi['score_dict']
        if score_dict['element_score'] > 0:
            return True
        elif score_dict['score_chemo'] > 0 and score_dict['score_chemo'] < 6 and score_dict['score_compounds'] > 0:
            return True                   

    return False

def _is_true_product(reaction_smi):
    rxn = reaction_smi['rxn']
    comp_list = reaction_smi['compounds']
    reagent_count = 0
    product_count = 0
    product_yield = False
    for comp in comp_list:
        if comp['compound_role'] in ['REAGENT', 'CATALYST', 'UNKNOWN', 'REACTANT']:
            reagent_count += 1
        elif comp['compound_role'] == 'PRODUCT':
            product_count += 1
            if 'digit_amounts' in comp and 'mol%' in comp['digit_amounts']:
                if comp['digit_amounts']['mol%'] > 0:
                    product_yield = True
                    
    if (rxn.countProducts() == 1 and 
        product_count == 1 and
        rxn.countReactants() == 0 and
        reagent_count == 0 and 
        product_yield):
        return True  #this is the case when only product was found without reactants. 
    

def _transform_reaction_combs(reaction, compid_descr, code2compid, compid_to_name):
    #объединяет в себе набор функций для превращения результата _build_reaction в хемоинформатический виды
    resulting_reaction, react_cands = _parse_reaction_comp_lists (reaction,  
                                                                     compid_descr, code2compid, compid_to_name)
    new_react_cands = [] 
    for clean_compounds in react_cands:
        new_react_cands.append(_initial_hole_fill(clean_compounds))
    processed_reaction, score, rxn, scale, score_dict = _analyze_reaction_list(new_react_cands)
    
    resulting_reaction['compounds'] = processed_reaction
    resulting_reaction['score'] = score
    resulting_reaction['smiles'] = rxn.smiles()
    resulting_reaction['rxn'] = rxn
    resulting_reaction['scale'] = scale
    resulting_reaction['score_dict'] = score_dict
    resulting_reaction['dist_before'] = reaction['dist_before']
    return resulting_reaction