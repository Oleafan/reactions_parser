import os
import re
import fitz
from collections import Counter
import nltk
import brutt_rec as br
import pandas as pd
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('punkt')



from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import gensim.downloader as api
# model = api.load('word2vec-google-news-300')
model = api.load('word2vec-google-news-300')

from  indigo import Indigo
indigo = Indigo()
indigo.setOption('molfile-saving-mode', 3000)

from dictionary import del_list, spec_types, nmr_atom_types, metal_name_list, waste_list, solv_smiles2name, not_useful_smiles, metal_complex_list, elements

def pdf2text_image(file_list, temp_folder):
    #file_list -  список путей к файлам, которые надо работать. Предполагается, что дои известен заранее, и его искать не надо. 
    
    text = ""
    num_pages = 0
    for file_path in file_list:
        #читает html, txt и pdf
#         print(file_path) 
        f_name = file_path.split('/')[-1]
        if file_path.endswith('.pdf'): #основная масса
            try:
                with fitz.open(file_path) as doc:
                    num_pages += len(doc)
                    for page_no, page in enumerate(doc):
                        text += page.get_text()
                        pixmap = page.get_pixmap(dpi=600)
                        pixmap.save(os.path.join(temp_folder, f_name + '_' +str(page_no) + '.png'))
            except Exception as e:
                print('Can not open file', file_path)
        elif '.txt' in file_path: #перегнали вордовские документы в txt
            try:
                with open(file_path, 'r', encoding = 'utf-8') as f:
                    text += f.read()
            except Exception as e: 
                print('Can not open file', file_path)
    #найти повторяющиеся строки в тексте. Если число повторений примерно = число страниц - выкинуть 
    text_splitted = text.split('\n')
    dict_text=dict(Counter(text_splitted))
    
    #убирать колонтитулы. Актуально для ситуаций, когда только один пдф. 
    if num_pages > 5:
        for string in dict_text.keys():
            if dict_text[string] == num_pages and len(string) > 5:
                text = text.replace('\n'+string, ' ')
                
    return text


def clean_exp_sec(text):
    #удалить номера страниц в формате 1 2 3 и s1 S2 s3:
    text = re.sub(r'\n\s*[S|s]?\s?\d+\s*\n', ' ', text)
    
    #удалить всякие странные символы, которые точно не имеют отношение к свойствам
    text = text.replace("¢", "'")
    text = text.replace("′", "'")
    text = text.replace("’", "'")
    text = text.replace('-DD-', '-D-') #This is one of the bugs in fitz pdf recognition.  
    text = text.replace('Η','H') #какой-то странный символ H, который херит весь парсинг протонных спектров
    text = text.replace('Н','H') #еще один странный символ H, который херит весь парсинг протонных спектров
    text = text.replace('$','')    
    text = text.replace('，', ', ')
    text = text.replace('；', ';')
    text = text.replace('\uf0b0C', '°C')
    text = text.replace('μ', 'µ')
    text = text.replace('°C', '°C')
    text = text.replace('\u00b0C', '°C')
    text = text.replace('℃', '°C')
    text = text.replace('×', 'x')
    text = text.replace('x', 'x')
    text = text.replace('\uf0b4', 'x')
    text = text.replace(' x ', 'x')
    text = text.replace('µ', 'µ')
    text = text.replace('buthyl', 'butyl')
    text = text.replace('l %', 'l%')
    text = text.replace('µ L', 'µL')
    text = text.replace('\u00b5', 'µ')
    text = text.replace('\uf06d', 'µ')
    text = text.replace(')palladium', ') palladium')
    text = text.replace(' bromine', ' dibromine')
    text = text.replace(' iodine', ' diiodine')
    
    text = re.sub(r'\bTMS\s*\n', ' ', text) #При парсинге пдф в текст часто вылезают куски формул. ТМС может втыкаться в идущие следом названия.  
    
    molar_conc = re.findall(r'\s\d+\s[NM]\s', text) #убрать что-то вида ' 1 N ', преобразовать в ' 1N '
    for item in molar_conc:
        text = text.replace(item, ' ' + item.replace(' ', '') + ' ')
    
    #удалить всякое стереохимическое левое
        
    stereo_list = ['(+)-', '(-)-', '()-', '(−)-', '(+)', '(-)', '()', '(−)' ]
    for stereo in stereo_list:
        text = text.replace(stereo, '') 
        
    sep_list = ['', '\0', '\a', '\b', '\t', '\n', '\v', '\f', '\r'] 
    for sep in sep_list: 
        text = text.replace(sep, ' ') #эти разделители тут не в тему. 
    
    #варианты дефисов
    text = text.replace('−', '-') 
    text = text.replace('–', '-')
    text = text.replace('–', '-')     
    text = text.replace('', '')
    text = text.replace('•', '*')
    text = text.replace('⋅', '*')
    text = text.replace('·', '*')
    text = text.replace('∙', '*')
    text = text.replace('\uf06c', 'lambda')
    text = re.sub(r'[Rr]hodium\s+[Aa]cetate', 'dirhodium (II) acetate', text)

    text = re.sub(r'[iI]\.[dD]\.', 'ID', text)
    
    #поставить пробел перед кодами соединений в скобках: 
    subst_code_regex_list = [ r'\(\d+[a-zABD-GI-Z]+[\'\’]*\)',  # (1a)
    r'\([a-zABD-GI-Z]{1,2}[\'\’]*\)',  # (a)
       r'\(\d{1,2}[\'\’]*\)',  # (1)
       r'\([a-zABD-GI-Z]+\d+[\'\’]*\)',  # (a1)
       r'\([a-zA-Z]{1,5}\-\d+[a-zABD-GI-Z]+[\'\’]*\)']  # (anti-1a)
       
    for code_regex in subst_code_regex_list:
        codes = re.findall(code_regex, text)
        for code in codes: 
            text = text.replace(code, ' ' + code)
               
    text = re.sub(r'\s+', ' ', text)
    
    #убрать переносы
    #любые 2 буквы в нижнем регистре + дефис + пробел + еще две буквы в нижнем регистре
    regex1 = r'[a-z]{2}-\s[a-z]{2}'
    #найти список таких вот строк, проверить, есть ли аналогичные слова без дефиса и с дефисом конкретно в этом тексте

    for string in list(set(re.findall(regex1, text))):
        wo_space = string.split(' ')[0]+string.split(' ')[1]
        wo_defis = string.split('- ')[0]+string.split('- ')[1]
        regex_temp_wo_space = r'\w+' + wo_space + '\w+'
        regex_temp_wo_defis = r'\w+' + wo_defis + '\w+'

        if len(re.findall(regex_temp_wo_space, text)) > len(re.findall(regex_temp_wo_defis, text)):
            text=text.replace(string, wo_space)
        else:
            text=text.replace(string, wo_defis)      
    
    list_decoupled = ['{'+atom + '}' for atom in nmr_atom_types] + ['{1H}', '{1 H}', '{19F}', '{13C}', '{11B}', '{31P}', '{15N}', '{77Se}', '{7Li}', '{27Al}', '{29Si}', '{17O}'] + ['{3Н}', '{3He}', '{10B}', '{23Na}', '{25Mg}', '{33S}', '{35Cl}', '{37Cl}', '{39K}', '{43Ca}', '{51V}', '{59Co}', '{65Cu}', '{67Zn}', '{79Br}', '{81Br}', '{83Kr}', '{87Rb}', '{111Cd}', '{113Cd}', '{127I}', '{129Xe}', '{133Cs}', '{195Pt}', '{199Hg}', '{205Tl}', '{207Pb}']
    
    for dec in list_decoupled:
        text = text.replace(dec, '')

    #унификация написания растворителей и характеристик
    regex_repl = {
        '1[\\s]?H.{0,2}NMR': ' H_NMR ',
        'NMR\\s*1H': ' H_NMR ',
        '[Pp]roton\\s+[Nn]uclear\\s+[Mm]agnetic\\s+[Rr]esonance\\s+[Ss]pectrum': ' H_NMR ',
        '.H\\sNMR': ' H_NMR ',
        'H.{0,2}NMR': ' H_NMR ',
        'δ1H': ' H_NMR ',
        '13[\\s]?C.{0,2}NMR': ' C_NMR ',
        'NMR\\s*13C': ' C_NMR ',
        'C.{0,2}NMR': ' C_NMR ',
        '[Cc]arbon\\s+[Nn]uclear\\s+[Mm]agnetic\\s+[Rr]esonance\\s+[Ss]pectrum': ' C_NMR ',
        'δ13C': ' C_NMR ',
        '13C\\s?\\(': ' C_NMR (',
        '[Ii]nfrared\\s+[Aa]bsorption\\s+[Ss]pectrum': ' I_R ',
        'Infrared': ' I_R ',
        'FTIR': ' I_R ',
        'FT IR': ' I_R ',
        ' IR': ' I_R ',
        'FT-IR': ' I_R ',
        ';IR': ' I_R ',
        '\\,IR': ' I_R ',
        'ν.?max': ' I_R ',
        'ATR\\.?IR': ' I_R ',
        '\\.IR': ' I_R ',
        'HR.{0,4}MS': ' HR_MS ',
        'HR\\s?-\\s?FAB\\s?MS': ' HR_MS ',
        'HR.?LC\\s?MS': ' HR_MS ',
        'High-Resolution.{0,20}MS': ' HR_MS ',
        '[Ee]xact\\s+[Mm]ass': ' HR_MS ',
        ' mp ': ' M_P ',
        '[Mm]\\.\\s?[Pp]\\.': ' M_P ',
        '\\s[M|m][P|p]\\s': ' M_P ',
        '11[\\s]?B.{0,2}NMR': ' B_NMR ',
        '31[\\s]?P.{0,2}NMR': ' P_NMR ',
        '31P\\s?\\(': ' P_NMR (',
        '19[\\s]?F.{0,2}NMR': ' F_NMR ',
        'δ19F': ' F_NMR ',
        '19F\\s?\\(': ' F_NMR (',
        '15[\\s]?N.{0,2}NMR': ' N_NMR ',
        '29[\\s]?S[iI].{0,2}NMR': ' Si_NMR ',
        '17[\\s]?O.{0,2}NMR': ' O_NMR ',
        'HPLC': ' HP_LC ',
        'SFC': ' HP_LC ',
        '[Dd][Mm][Ss][Oo]\\s?-\\s?[dD]\\s?6': 'DMSO_d6',
        'd6\\s?-\\s?[Dd][Mm][Ss][Oo]': 'DMSO_d6',
        '\\(CD3\\)2SO': 'DMSO_d6',
        'CDCl\\s?3': 'CDCl_3',
        '[Cc]hloroform\\s?-?\\s?d': 'CDCl_3',
        'CDC\\s*l\\s*3': 'CDCl_3',
        'CDC13': 'CDCl_3',
        'd\\s?-?\\s?[Cc]hloroform': 'CDCl_3',
        'CD3OD': 'CD3_OD',
        '[Mm]ethanol\\s?-\\s?d4': 'CD3_OD',
        'CD\\s?3.{0,2}OD': 'CD3_OD',
        'MeOH.{0,2}d[34]': 'CD3_OD',
        'd[34]\\s?-?\\s?[Mm]ethanol': 'CD3_OD',
        'MeOD': 'CD3_OD',
        'CD3CN': 'CD3_CN',
        '[Aa]cetonitrile\\s?-?\\s?d3': 'CD3_CN',
        'd3\\s?-?\\s?[Aa]cetonitrile': 'CD3_CN',
        '[Aa]cetone\\s?-?\\s?d6': '(CD_3)_2CO',
        '\\(CD3\\)2CO': '(CD_3)_2CO',
        'd.?[Aa]cetone': '(CD_3)_2CO',
        'd6\\s?-?\\s?[Aa]cetone': '(CD_3)_2CO',
        'CD3COCD3': '(CD_3)_2CO',
        'D\\s?2\\s?O': 'D2_O',
        'water\\s?-?\\s?d2': 'D2_O',
        'd2\\s?-?\\s?water': 'D2_O',
        'br. s': 'br_s',
        'br.s': 'br_s',
        'broad s': 'br_s',
        'THF\\s?-?\\s?d8': 'THF_d8',
        'd8\\s?-?\\s?THF': 'THF_d8',
        '[Cc][Dd]\\s?2\\s?[Cc]l\\s?2': 'CD2_Cl2',
        '[Dd]ichloromethane.{0,3}d2': 'CD2_Cl2',
        'd2.{0,3}[Dd]ichloromethane': 'CD2_Cl2',
        'C6D6': 'C6_D6',
        '[Bb]enzene.{0,3}d6': 'C6_D6',
        'd6.{0,3}[Bb]enzene': 'C6_D6',
        '[Tt]oluene\\s?-?\\s?d8': 'toluene_d8',
        'd8\\s?-?\\s?[Tt]oluene': 'toluene_d8',
        'd8\\s?-?\\s?tol': 'toluene_d8',
        'C7D8': 'toluene_d8',
        'CD\\s?3\\s?NO2': 'CD3_NO2',
        '[Nn]itromethane\\s?\\-?\\s?d\\s?\\-?\\s?3': 'CD3_NO2',
        'd\\s?\\-?\\s?3\\s?\\-?\\s?[Nn]itromethane': 'CD3_NO2',
        'TFE\\s?-?\\s?d3': 'TFE_d3',
        'C6D5Cl': 'C6D5_Cl',
        '[Cc]lorobenzede\\s?-?\\s?d5': 'C6D5_Cl',
        'd5\\s?-?\\s?[Cc]lorobenzede': 'C6D5_Cl',
        'silica gel': 'silica_gel',
        '77[\\s]?Se.{0,2}NMR': ' Se_NMR ',
        '7[\\s]?:Li.{0,2}NMR': ' Li_NMR ',
        '27[\\s]?Al.{0,2}NMR': ' Al_NMR ',
        '\\[a\\][dD]': 'opt_rotation ',
        '\\[α\\][dD]?': 'opt_rotation ',
        '[\uf061]': 'opt_rotation ',
        '[\ue3b1]': 'opt_rotation ',
        '½a�': 'opt_rotation ',
        '\\[a\\]\\d+': 'opt_rotation ',
        '[Cc]hiral\\s?GC': ' Chiral_GC ',
        ' GC ': ' Chiral_GC ',
        '\uf02d': '-',
        '\x02': '-',
        '�': '-',
        r'app\.': 'app ',
        '[Cc]alcd.': 'calcd',
        '\\b[Bb]r\\.': 'br_',
        'H\\s*arom\\.': 'Harom',
        'C\\s*arom\\.': 'Carom',
        '\\sarom\\.': ' arom',
        '[Ee]strone': '(3aS,3bR,9bS,11aS)-7-Hydroxy-11a-methyl-2,3,3a,3b,4,5,9b,10,11,11a-decahydro-1H-cyclopenta[a]phenanthren-1-one'
    }

    for pattern in regex_repl:
        text = re.sub(pattern, regex_repl[pattern], text)

    dict_nmr_regex = {}
    for atom in nmr_atom_types:
        dict_nmr_regex.update({' ' + atom+'_NMR ' : r'\d{1,3}\s?'+atom + r'[^\)]{0,2}NMR'}) # '.{0,2}NMR'}) ( (340 K) NMR (C6D6) становится K_NMR) 

    for key in dict_nmr_regex:
        text = re.sub(dict_nmr_regex[key], key, text)
    
    #в одном случае обнаружил, что дельта после ЯМР распозналась как !. C_NMR   (100 MHz, DMSO_d6, 293K): ! 164.8, 164.4, 134.4, 132.4, 131.0, 129.5, 129.2, 128.7, 127.4, 126.9
    #попытка это вылечить: удаляем все восклицательные знаки в пределах 50 символов после обозначения какого-либо спектра. 
    
    for spec in spec_types: 
        regex = spec + r'.{0,50}!'
        found_strings = re.findall(regex, text)
        for string in found_strings:
            text = text.replace(string, string.replace('!', ''))
    
    extra_space_list = re.findall(r'\,\s+\d+\.\s\d{1,4}\s?\(', text) #', 7. 39 ('
    if len(extra_space_list) > 0:
        for spaced in extra_space_list:
            text = text.replace(spaced, spaced.replace('. ', '.'))
    extra_space_list = re.findall(r'\,\s+\d+\.\s\d{1,4}\-\d+\.', text) #', 7. 42-7.'
    if len(extra_space_list) > 0:
        for spaced in extra_space_list:
            text = text.replace(spaced, spaced.replace('. ', '.'))    
    extra_space_list = re.findall(r'\)\:\s+\d+\.\s\d{1,4}\,', text) #'): 162. 4,'
    if len(extra_space_list) > 0:
        for spaced in extra_space_list:
            text = text.replace(spaced, spaced.replace('. ', '.'))  
    extra_space_list = re.findall(r'δ\:?\s+\d+\.\s\d{1,4}\,', text) #'): 162. 4,'
    if len(extra_space_list) > 0:
        for spaced in extra_space_list:
            text = text.replace(spaced, spaced.replace('. ', '.')) 
            
    extra_point_list = re.findall(r'\([^\)]{0,6}[dtsm]\.', text) #'(dd. '. '(3H, d.' (в описании спектров - по этой точки сплитится предложение)
    if len(extra_point_list) > 0:
        for point in extra_point_list:
            text = text.replace(point, point[:-1])
    
    #print('close to split')
    #3+буквы в нижнем регистре и вплотную 1-3 цифры - это неправильная комбинация  (но Pd2dba3 - правильная 
    regex = r'[^\sA-Z\d][a-z]{3,}\d{1,3}'
    to_split = re.findall(regex, text)
    to_split.sort(key = len, reverse = True)
    for spl in to_split:
        new_spl = re.sub(r'\d', '',spl) 
        text = text.replace(spl, new_spl)
    
    #заменить всякое вида 3x30 ml на 3_x_30 ml
    washings = re.findall(r'\d+\s*x\s*\d+\s*mL', text)
    for wash in washings:
        text = text.replace(wash, wash.replace('x', '_x_'))
    
    #bicarbonate(4.0 mmol, 400 mg) может ломать распознавание молекул
    regex = r'\w\(\d+\.\d+\s' #'e(4.0 '
    items = re.findall(regex, text)
    for item in items:
        text = text.replace(item, item.replace('(', ' ('))
    
    #если перед malonate  стоит буква - добавить пробел
    malonates = re.findall(r'[a-z]malonate\s', text)
    for malonate in malonates:
        text = text.replace(malonate, malonate.replace('malonate', ' malonate'))
    
    #также добавить пробел перед металлами
    for metal in metal_name_list:
        search_regex = r'[a-z]' + metal +r'\s'
        for item in re.findall(search_regex, text):
            text = text.replace(item, ' ' + item)
    text = text.replace('/', ' / ')
    
    return text

def _is_trash(sent):
    if 'doi.org' in sent or 'acs.org' in sent or  'Wiley-VCH' in sent:
        return True
    
    if len(sent) > 40:
        return False
    splitted = sent.replace('-', ' ').replace('.', ' ').replace(',', ' ').split(' ')
    splitted = [x for x in splitted if len(x)> 0]
    if len(splitted) == 0:
        return True
    av_len = sum([ len(x) for x in splitted])/len(splitted)
    if av_len <= 3:
        return True
    num_and_cap_fraction = (sum(1 for c in sent if c.isupper()) + sum(1 for c in sent if c.isnumeric()))/len(sent)
    if num_and_cap_fraction > 0.3:
        return True
    if len(sent) < 2:
        return True
    return False

def tokenize_exp_sec(exp_sec):
    try: 
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')
        
    hrms_regex = r'HR_MS.{1,60}\b\d{1,4}\.\d{4,5}\b.{1,30}\b\d{1,5}\.\d{4,5}[\.|,|;]*'
    
    hrms_list = re.findall(hrms_regex, exp_sec)
    for hrms in hrms_list:
        exp_sec = exp_sec.replace(hrms,'%SEP%'+hrms+'%SEP%')
    
    exp_sec_splitted = exp_sec.split('%SEP%')
    
    exp_sec_tokenized = []
    for part in exp_sec_splitted:
        
        if len(re.findall(hrms_regex, part)) > 0: 
            exp_sec_tokenized.append(part)
            
        else:
            exp_sec_tokenized+=list(nltk.sent_tokenize(part)) #разбить по предложениям
    
    #проверка того, что все характеристики начинаются с новой строки. 
    #В целом все спектры обозначаю с нижним подчеркиванием
    
    exp_sec_tokenized_new = []
    
    for sent in exp_sec_tokenized:
        for spec in spec_types: 
            pars_sent_list = re.findall(spec, sent) #найти все названия спектров (ЯМР и строковых) в токене
            for pars_sent in pars_sent_list: 
                sent = sent.replace(pars_sent, '%SEP%' + pars_sent)
        for token in sent.split('%SEP%'):
            exp_sec_tokenized_new.append(token)
    
    exp_sec_tokenized = exp_sec_tokenized_new.copy()  
    #к данному моменту у нас должен быть список предложений, и названия свойств могут быть только в начале предложения
       
    exp_sec_tokenized_new = []
    #check if there any issues when two properties are in one sent with NMR spec : (речь идет про ситуации, когда возникают какие-то еще свойства, которые парсилка не парсит)
    parse_list = [r'\s+Rf', r'[^D|^T]MS', r'm\s*/\s*z', r'[M|m]ass', 'λ', 'MALDI', r'[C|c]alcd', 'Spectral']
    for sent in exp_sec_tokenized:
        if 'NMR' in sent or 'I_R' in sent:  #HP_LC сюда не надо добавлять, так как тогда излишне разломаются предложения "HP_LC  (OD-3), n-heptane/i-PrOH 95:5, 1.0 ml/min, λ = 210 nm, tmajor = 9.38 min, tminor = 8.78 min; er = 97.5:2.5. "
            for parser in parse_list:
                pars_sent_list = re.findall(parser, sent)
                for pars_sent in pars_sent_list: 
                    sent = sent.replace(pars_sent, '%SEP%' + pars_sent)
            if 'NMR' in sent:
                pars_sent_list = re.findall(r'[A-Z][a-z]{1,20}\,\s?[A-Z]\.', sent) #фамилии авторов
                for pars_sent in pars_sent_list: 
                    sent = sent.replace(pars_sent, '%SEP%' + pars_sent)
            for token in sent.split('%SEP%'):
                exp_sec_tokenized_new.append(token)
        else:
            exp_sec_tokenized_new.append(sent)

    return [x for x in exp_sec_tokenized_new if not _is_trash(x)] 

def _digit_score(text):
    #Function calculates the fraction of numeric values in the text string
    numer = 0
    for symb in text:
        try: 
            int(symb)
            numer +=1
        except:
            pass
    return numer/len(text)

def get_candidates_whole(exp_sec, n_grams = 6):
    #n_grams - это количество слов, которое может быть в химическом названии
    cand_list = []
   
    for sent in exp_sec:
        if len(sent)<5:
            continue #too short name
        sent = sent.replace(' and ', ' _and_ ')
        

        words = sent.split()
        words = [w for w in words if w not in waste_list]
        if len(words) > 0:
            max_len = len(max(words, key=len))
        else:
            max_len = 0
        sent_new_temp = ' '.join(words)

        if max_len < 5:
            continue #too short elements of the sent

        if len(sent_new_temp)<5:
            continue #too short name

        if len(words) > 1:
            for grams in range(n_grams,0,-1):
                for start_index in range(0, len(words)-grams+1):
                    combined = ' '.join(words[start_index: start_index+grams])
                    if _digit_score(combined) < 0.4:
                        cand_list.append(combined)
                        if '(E/Z)-' in combined:
                            cand_list.append(combined.replace('(E/Z)-', ''))
        else:
            cand_list.append(sent_new_temp)
            if '(E/Z)-' in sent_new_temp:
                cand_list.append(sent_new_temp.replace('(E/Z)-', ''))
    
    cand_list = [x.strip() for x in cand_list]
    
    cand_list_new = []
    for cand in cand_list:

        if cand[-1] in ['.', ',', ';', '-', '/', '|', '\\', '(', '[']:
            cand_list_new.append(cand[:-1])

            #надо как-то учесть случаи вида "peroxybis(triethylsilane)3,7", 
        # то есть когда в конце список чисел, которые сюда особо не в тему
        ref_cands = re.findall(r'(?:\d+\,)+\d+', cand)
        if len(ref_cands) > 0:
            if cand[len(cand) - len(ref_cands[-1]):] == ref_cands[-1]:
                cand_list_new.append(cand[:len(cand) - len(ref_cands[-1])])
            
            
        #проверка, что в конце cand стоит какое-то число или число в квадртаных скобках 
        #типа acetone1 or acetone[1]. Эта ситцация указывает на то, что ссылка на литературу 
        #слилась к именем компаунда
        if cand[-1].isnumeric(): 
            cand_list_new.append(cand[:-1])
        if cand[-1] == ']':
            ref_cands = re.findall(r'\[\d+[^\]]{0,3}\]', cand)
            if len(ref_cands) > 0:
                if cand[len(cand) - len(ref_cands[-1]):] == ref_cands[-1]:
                    cand_list_new.append(cand[:len(cand) - len(ref_cands[-1])])
                
        if '(SR)-' in cand: 
            cand_list_new.append(cand.replace('(SR)-', ''))
        
        cand_list_new.append(cand)
       
    #cand_list = [x.replace('.', '') for x in cand_list_new]  #так нельзя ибо [2.2.1] в названиях может быть
    cand_list = [x.strip() for x in cand_list_new]
    cand_set = list(set(cand_list)) #remove duplicates
    cand_set.sort(key = len, reverse=True) #sorting by decreasing of the lenth of the candidate
    
    return cand_set

def _get_indigo_charge_radicals(mol):
    ch = 0
    rad = 0
    for atom in mol.iterateAtoms():
        ch += atom.charge()
        rad += atom.radicalElectrons()
    return ch, rad 

def get_name_to_smi (folder):
    #function takes the list of IUPAC names and returns a dictionary of IUPAC_NAME:Smiles
    input_file = os.path.join(folder, 'candidates.txt')
    output_smiles_file = os.path.join(folder, 'smiles.txt')

    #создаю словарь name_to_smi, в котором токены текста приводятся в соответствие со смайлсами. 
    name_to_smiles, solv_dict, misc_dict = {}, {}, {}
    solvents_smiles = list(solv_smiles2name.keys())
    
    with open(output_smiles_file, 'r', encoding="utf-8") as f_smiles, open(input_file, 'r', encoding = 'utf-8') as f_candidates:
        while True:
            candidate = f_candidates.readline()[:-1]
            if len(candidate) == 0:
                break

            smiles = f_smiles.readline()[:-1]
            if (len(smiles) > 0) : 
                if smiles in solvents_smiles:
                    #this is solvent
                    solv_dict.update({candidate: smiles})
                else:
                    try:
                        if smiles not in not_useful_smiles:
                            mol = indigo.loadMolecule(smiles)
                            charge, radicals = _get_indigo_charge_radicals(mol)
                            
                            if charge == 0 and radicals == 0:
                                name_to_smiles.update({candidate: smiles}) 
                            elif charge is not None:
                                for metal in metal_complex_list:
                                    if metal in smiles:
                                        name_to_smiles.update({candidate: smiles}) 
                                        break
                                
                    except Exception as e:
                        print('get_name_to_smi', e)
                        pass
            else:
                try:
                    string = candidate
                    if len(string) < 35:
                        smiles = br.recognize_string(string)
                        if smiles is not None:
                            misc_dict.update({string: smiles})
                except:
                    pass
    key_to_drop = []
    for key, smiles in misc_dict.items():
        if not is_adequate_mol(smiles):
            key_to_drop.append(key)
            print('wrong_smiles', key, smiles)
    for key in key_to_drop:
        misc_dict.pop(key, None)
    
    
    return name_to_smiles, solv_dict, misc_dict

def is_adequate_mol(smiles):
    try:
        mol = indigo.loadMolecule(smiles)
        if mol:
            return True
        else:
            return False
    except:
        return False

def get_spec_list(exp_sec_tokenized, name_to_smiles):
    
    name_to_smi_bordered = {k: '%SMILES%%SMI%'+v+'%SMI%%SMILES%' for k,v in name_to_smiles.items()}
    
    exp_sec_smi = []
    for sent in exp_sec_tokenized:
        if 'HP_LC' in sent: #check is there HP_LC description in the sentence? В этом случае ничего делать не надо.
            exp_sec_smi.append(sent)
        else:
            #в каждом предложении сначала заменяем токены на инчи или на смайлсы.
            sent_new = sent
            for name in name_to_smi_bordered:
                if name in sent:
                    sent_new = sent_new.replace(name,name_to_smi_bordered[name])
            exp_sec_smi.append(sent_new)
            
    #у нас получился список предложений. В каждом предложении все названия по ИЮПАК, что были, заменены на ИНЧИ/смайлсы. 
    #tokenize by inchi and smiles code
    exp_sec_clean = []
    for sent in exp_sec_smi:
        tokens = sent.split('%SMILES%')
        for tok in tokens:
            if len(tok) > 0:
                exp_sec_clean.append(tok)       


    exp_sec = [{'sentence': sentence} for sentence in exp_sec_clean]

    #классификация что из себя представляет предложение
    for sent in exp_sec:
        if '%SMI%' in sent['sentence']:
            sent['sent_type'] = 'smiles'
        else: 
            check_spec = False
            for spec in spec_types:
                if spec in sent['sentence']:
                    check_spec = True
            if check_spec:
                if _digit_score(sent['sentence'])>0.15:
                    sent['sent_type'] = 'spec' 
                elif 'HP_LC' in sent['sentence'] and _digit_score(sent['sentence'])>0.05:
                    sent['sent_type'] = 'spec'
                elif 'Chiral_GC' in sent['sentence'] and _digit_score(sent['sentence'])>0.05:
                    sent['sent_type'] = 'spec'
                elif 'HR_MS' in sent['sentence'] and re.search(r'\d+\.\d{4}', sent['sentence']):
                    sent['sent_type'] = 'spec'
                elif 'opt_rotation' in sent['sentence'] and _digit_score(sent['sentence'])>0.05:
                    sent['sent_type'] = 'spec'
                else: 
                    sent['sent_type'] = 'trash'
            else:
                sent['sent_type'] = 'trash'

    return [sent['sentence'] for sent in exp_sec if sent['sent_type'] == 'spec']


def segmentate_exp_sec(exp_sec_new, spec_list):

    from chemdataextractor.doc import Paragraph
    
    for spec in spec_list:
        if 'M_P' in spec:
            exp_sec_new = exp_sec_new.replace(spec, ' ')
        elif 'HP_LC' not in spec:
            exp_sec_new = exp_sec_new.replace(spec, ' %SPEC% ')


    #Растянуть элементы вида 'Data consistent with that reported in the literature.1'
    spl = re.findall(r'[A-Za-z]\.\d', exp_sec_new)
    for sp in spl:
        exp_sec_new = exp_sec_new.replace(sp, sp.replace('.', '. '))

    splitted = exp_sec_new.replace('%SPEC%', '\n').split('\n')
    
    #всякие словари для сплита
    list_of_yield_marks = ['yield', 'afford', 'furnish']
    regexes_yield_strong = [r'provided.{1,10}desired']
    regexes_yield = [r'was\s+purified', r'was\s+isolated', r'is\s+purified', r'is\s+isolated',
                    r'to\sobtain.{1,20}pure', r'was\s+obtained', 'to\s+obtain', 'to\sgive']
    capital_titles = [r'EXPERIMENTAL\sPROCEDURES', r'CHARACTERIZATION\sDATA', r'[Cc]haracterization\s[Dd]ata']
    
    starts_regex = [r'[Ss]ynthetic\s[Pp]rocedures', r'SYNTHETIC\sPROCEDURES', '[Ss]ynthesis\sof\sstarting\ssubstrates', r'EXPERIMENTAL\sPROCEDURES', r'CHARACTERIZATION\sDATA', r'[Cc]haracterization\s[Dd]ata', r'[Ss]ynthetic\s[Aa]pplications']
    
    comm_available = ['are\s+commercially\s+available', 'is\s+commercially\s+available']
    
    #for checking using Word Mover's Distance
    sents_finish = ['Data consistent with that reported in the literature',
                    'NMR spectra are in accrodance with the literature data',
                    'The spectroscopic properties of this compound matched those reported in the literature',
                    'in agreement with those previously described']
    sents_finish = [[w.lower() for w in x.split() if w.lower() not in stop_words] for x in sents_finish] 
    
    sents_start = ['Synthetic Procedures', 'Reactions protocols', 'Synthesis of Starting Materials', 'Characterization of Aryl Nitriles in Table', 'Characterization of Heteroaryl Nitriles in Table', 'Characterization of compounds']
    sents_start = [[w.lower() for w in x.split() if w.lower() not in stop_words] for x in sents_start] 
    
    
    for item in splitted:
        para = Paragraph(item)

        yield_found = False
        buffer = 0 
        for sent in para.sentences[::-1]:

            if re.search(r'MS\s+\(?EI\)?', sent.text) and _digit_score(sent.text) > 0.17:
                exp_sec_new = exp_sec_new.replace(sent.text, ' %SPEC% ' )
                continue

            if re.search(r'm\s?\/\s?z', sent.text) and _digit_score(sent.text) > 0.17:
                exp_sec_new = exp_sec_new.replace(sent.text, ' %SPEC% ' )
                continue
            
            sent_text = sent.text.lower() 
            
            for regex in comm_available: #наличие этих токенов - точно надо сплитить, это предложение не может быть методикой. 
                if re.search(regex, sent_text):
                    exp_sec_new = exp_sec_new.replace(sent.text, ' %SEP% ' + sent.text + ' %SEP% ' )
            
            for mark in list_of_yield_marks: #наличие этих токенов - точно надо сплитить. 
                if mark in sent_text:
                    exp_sec_new = exp_sec_new.replace(sent.text, sent.text + ' %YI%%SEP% ' )
                    yield_found = True
                    
            for regex in regexes_yield_strong:
                if re.search(regex, sent_text):
                    exp_sec_new = exp_sec_new.replace(sent.text, sent.text + ' %YI%%SEP% ' )
                    yield_found = True
            #если выход уже найден - следующее предложение (листаем с конца) не должно получить метку выхода
            if not yield_found:
                for regex in regexes_yield:
                    if re.search(regex, sent_text):
                        exp_sec_new = exp_sec_new.replace(sent.text, sent.text + ' %YI%%SEP% ' )
                        yield_found = True
            else:
                buffer += 1
                if buffer > 1:
                    yield_found = False
                    buffer = 0
            
            
            sent_text = re.sub(r'the\s+general\s', 'the_general_', sent_text)
            sent_text = re.sub(r'using\s+general\s', 'using_general_', sent_text)
            sent_text = re.sub(r'following\s+general\s', 'following_general_', sent_text)
            sent_text = re.sub(r'to\s+general\s', 'to_general_', sent_text)
            sent_text = re.sub(r'from\s+general\s', 'from_general_', sent_text)
            sent_text = re.sub(r'general\s+procedure.{1,8}was\sfollowed', 'general_procedure_was_followed_', sent_text)
            sent_text = re.sub(r'typical\s+procedure.{1,8}was\sfollowed', 'typical_procedure_was_followed_', sent_text)

            sent_text = re.sub(r'the\s+typical\s', 'the_typical_', sent_text)
            sent_text = re.sub(r'using\s+typical\s', 'using_typical_', sent_text)
            sent_text = re.sub(r'following\s+typical\s', 'following_typical_', sent_text)
            sent_text = re.sub(r'to\s+typical\s', 'to_typical_', sent_text)
            sent_text = re.sub(r'from\s+typical\s', 'from_typical_', sent_text)

            if re.search(r'\bgeneral\s+procedure', sent_text):
                exp_sec_new = exp_sec_new.replace(sent.text, ' %SEP%%GP% ' + sent.text)
            if re.search(r'\bgeneral\s+protocol', sent_text):
                exp_sec_new = exp_sec_new.replace(sent.text, ' %SEP%%GP% ' + sent.text)

            if re.search(r'\btypical\s+procedure', sent_text):
                exp_sec_new = exp_sec_new.replace(sent.text, ' %SEP%%GP% ' + sent.text)
            if re.search(r'\btypical\s+protocol', sent_text):
                exp_sec_new = exp_sec_new.replace(sent.text, ' %SEP%%GP% ' + sent.text)

            sent_gens = [w.lower() for w in sent_text.split() if w.lower()  not in stop_words]
            if 'data' in sent_gens or 'spectra' in sent_gens or 'reported' in sent_gens:
                finish_dists = [model.wmdistance(x, sent_gens) for x in sents_finish]
                if min(finish_dists) < 1: 
                    exp_sec_new = exp_sec_new.replace(sent.text, sent.text + ' %REF%%SEP% ' )
            
            start_dists = [model.wmdistance(x, sent_gens) for x in sents_start]

            if min(start_dists) < 1: 
                exp_sec_new = exp_sec_new.replace(sent.text, ' %SEP%%SP% ' + sent.text  )
                
            for mark in capital_titles:
                if re.search(mark, sent.text) :
                    exp_sec_new = exp_sec_new.replace(sent.text, ' %SEP%%GP% ' + sent.text )
                    
            for mark in starts_regex:
                found_marks = re.findall(mark, sent_text)
                if len(found_marks) > 0:
                    exp_sec_new = exp_sec_new.replace(sent.text, re.sub(mark, ' %SEP%%SP% ' +  found_marks[0], sent.text) )


    while exp_sec_new.count('  ') > 0:
        exp_sec_new = exp_sec_new.replace('  ', ' ')
    
    #обработка кусков вида %REF%%SEP% 14 и %REF%%SEP% [14]
    exp_sec_new = re.sub(r'\%REF\%\%SEP\%\s*\d+\s+', '%REF%%SEP% ', exp_sec_new)
    exp_sec_new = re.sub(r'\%REF\%\%SEP\%\s*\[\d+]\s+', '%REF%%SEP% ', exp_sec_new)
    
    #удаление кусков вида bla bla. 14 Bla bla 
    replace_els = re.findall(r'[a-z]{4}\.\s*\d+\s+[A-Z][a-z]{2}', exp_sec_new) 
    for replace_el in replace_els:
        exp_sec_new = exp_sec_new.replace(replace_el, re.sub(r'\d', ' ', replace_el))
        
        
    return exp_sec_new


def remove_extra_smiles(name_to_smi):
    #если в молекулу вдруг вошло два независимых фрагмента, и оба фрагмента нейтральны и обозначены ниже - удалить нафиг, и ни одна молекула не является кислотой
    name_to_indigo_smiles = {}
    for k,v in name_to_smi.items():
        name_to_indigo_smiles.update({k: indigo.loadMolecule(v).smiles()})
    indigo_smiles_to_name = {v:k for k, v in name_to_indigo_smiles.items()}
    
    do_not_consider_smiles = ['Cl', 'Br', 'I', 'F[B-](F)(F)F', 'F[P-](F)(F)(F)(F)F', 'C1(=CC=CC=C1)[B-](C1=CC=CC=C1)(C1=CC=CC=C1)C1=CC=CC=C1', 'Pd', '[Pd]']
    
    keys_to_remove = []
    smi_list = [v for k,v in name_to_indigo_smiles.items()]
    for key in name_to_indigo_smiles.keys():
        add_to_remove_list = False
        frag_smiles_list = []
        try:
            if '.' in name_to_indigo_smiles[key]:
                add_to_remove_list = True
                for smi in name_to_indigo_smiles[key].split('.'):
                    frag_smiles_list.append(smi)
                    
            for smi in frag_smiles_list:
                if smi not in smi_list:
                    add_to_remove_list = False
                    break

            for smi in frag_smiles_list:
                if smi in indigo_smiles_to_name.keys():
                    if indigo_smiles_to_name[smi] not in name_to_indigo_smiles.keys():
                        add_to_remove_list = False
                        break
            
            for smi in frag_smiles_list:
                if smi in do_not_consider_smiles:
                    add_to_remove_list = False
                    break
            for smi in frag_smiles_list:
                for metal in metal_complex_list:
                    if metal in smi:
                        add_to_remove_list = False
                        break
            
            if add_to_remove_list == False:
                continue

            if add_to_remove_list:
                keys_to_remove.append(key)
        except: 
            pass
    for key in keys_to_remove:
        name_to_smi.pop(key, None)
    return name_to_smi

def get_joint_substr_dict(name_to_smi, solv_dict, misc_dict):
    #то, что было в ИЮПАКе, и преобразовалось в smiles. Составить словарь. 
    #joint_sorted_dict - отсортированнный по длине токена словарь токен в тексте: универсальный токен типа Compound_0.
    #codes_desc - словарь, где каждому кодовому токену типа Compound_0 соответствует конкретный смайлс.
    
    #all solvents should be placed into solv_dict. Not misc_dict or name_to_smi
    solvent_mols = [indigo.loadMolecule(x) for x in solv_smiles2name]
    
    tokens_to_del = []
    for name, smiles in name_to_smi.items():
        mol = indigo.loadMolecule(smiles)
        for solv_mol in solvent_mols:
            if indigo.exactMatch(mol, solv_mol):
                tokens_to_del.append(name)
                solv_dict[name] = smiles
                break
    for token in tokens_to_del:
        name_to_smi.pop(token, None)
    
    tokens_to_del = []
    for name, smiles in misc_dict.items():
        mol = indigo.loadMolecule(smiles)
        for solv_mol in solvent_mols:
            if indigo.exactMatch(mol, solv_mol):
                tokens_to_del.append(name)
                solv_dict[name] = smiles
                break
    for token in tokens_to_del:
        misc_dict.pop(token, None)   
    
    subst_dict = {k.strip(): v for k,v in name_to_smi.items()}
    smi_list = list(set([v for k,v in subst_dict.items()]))

    smi_to_repr = {}
    for idx, smi in enumerate(smi_list):
        smi_to_repr.update({smi.strip(): 'Compound_' + str(idx)})
        
    subst_dict = {k.strip(): smi_to_repr[subst_dict[k]] for k in subst_dict}

    solv_br_d = {}
    for k, v in solv_smiles2name.items():
        for name in v:
            solv_br_d.update({name: k})
    solv_dict.update(solv_br_d)    
    
    solv_name_list = list(solv_dict.keys())
    solv_name_list.sort(key = len, reverse=True)
    sorted_solv_dict = {}
    for key in solv_name_list:
        sorted_solv_dict.update({key: solv_dict[key]})


    solv_smi_list = list(set([v for k,v in sorted_solv_dict.items()]))

    solv_smi_to_repr = {}
    for idx, smi in enumerate(solv_smi_list):
        solv_smi_to_repr.update({smi.strip(): 'Solvent_' + str(idx)})
    solv_dict_new = {}
    for key in sorted_solv_dict.keys():
        solv_dict_new.update({key.strip(): solv_smi_to_repr[sorted_solv_dict[key]]})

    
    misc_name_list = list(misc_dict.keys())
    misc_name_list.sort(key = len, reverse=True)
    sorted_misc_dict = {}
    for key in misc_name_list:
        sorted_misc_dict.update({key: misc_dict[key]})

    misc_inch_list = list(set([v for k,v in sorted_misc_dict.items()]))

    misc_inch_to_repr = {}
    for idx, inch in enumerate(misc_inch_list):
        misc_inch_to_repr.update({inch.strip(): 'MiscInch_' + str(idx)})

    misc_dict_new = {}
    for key in sorted_misc_dict.keys():
        misc_dict_new.update({key.strip(): misc_inch_to_repr[sorted_misc_dict[key]]})

    joint_subst_dict = {}
    joint_subst_dict.update(subst_dict)
    joint_subst_dict.update(solv_dict_new)
    joint_subst_dict.update(misc_dict_new)

    subst_sorted = list(joint_subst_dict.keys())
    subst_sorted.sort(key = len, reverse=True)

    joint_sorted_dict = {} #задача - отсортировать этот словарь. Наверное сделано не  оптимально, но работает.
    for key in subst_sorted:
        joint_sorted_dict.update({key: joint_subst_dict[key]})
    
    codes_desc =  {}
    codes_desc.update({v:k for k, v in smi_to_repr.items()})
    codes_desc.update({v:k for k, v in solv_smi_to_repr.items()})
    codes_desc.update({v:k for k, v in misc_inch_to_repr.items()})
    
    return joint_sorted_dict, codes_desc


def _replace_chem_tokens(exp_sec_new, joint_sorted_dict):
    #проводим замену токенов из subst_dict на идентификаторы. 
    #Надо, чтобы этот токен не являлся частью названия. То есть справа и слева от него должны быть пробелы/точка/запятая
    open_border_regex = r'[\s\.\,\;\:\/\[]\(?'
    close_border_regex = r'\)?[\s\.\,\;\:\/\d\[]' #в конце может быть цифра или открывающаяя квадратная/круглая скобка - это случаи вида acetone1. 1 - ссылка на статью. 
    specials = ['.', ',', '-', '[', ']', '(', ')', ';', '*', '%', '+']

    open_border_regex_solv = r'[\s\.\,\;\:\/\-]\(?'
    close_border_regex_solv = r'\)?[\s\.\,\;\:\/\d\[\-=]'


    for subst in joint_sorted_dict.keys():
        new_subst = subst
        for special in specials:
            new_subst =  new_subst.replace(special, '\\' + special)
        if 'Solvent_' in joint_sorted_dict[subst]:
            regex_to_find = open_border_regex_solv +new_subst + close_border_regex_solv
        else:
            regex_to_find = open_border_regex +new_subst + close_border_regex
        tokens_to_sub = re.findall(regex_to_find, exp_sec_new)

        for token in tokens_to_sub:
            exp_sec_new = exp_sec_new.replace(token, 
                                              token.replace(subst, '%COMP%'+ joint_sorted_dict[subst] + '%COMP%'))
    return exp_sec_new



def _remove_extra_spaces(sent):
    while sent.count('  ') > 0:
        sent = sent.replace('  ', ' ')
    return sent

def _get_repr_to_code_dict(exp_sec):
    """
    берет эксперименталку где проведены замены всех хим токенов на %COMP%Compound_XXX%COMP% 
    возвращает словарь инчи/смайлс:код соединения
    """
    exp_sec_new = ""
    exp_sec_new = exp_sec[:] #скопировать эксперименталку, чтобы исходный текст не херить
    
    while exp_sec_new.count('  ') > 0:
        exp_sec_new = exp_sec_new.replace('  ', ' ')
        
    exp_sec_new_smi =  exp_sec_new.split("%COMP%")
    
    
    subst_code_regex_1 = r'\(\d+[a-zABD-GI-Z]+[\'\’]*\)'  # (1a)
    subst_code_regex_2 = r'\s\d+[a-zABD-GI-Z]+[\'\’]*[\s\.]'  #1a, ограниченный пробелами
    subst_code_regex_3 = r'\([a-zABD-GI-Z]{1,2}[\'\’]*\)'  # (a)
    subst_code_regex_4 = r'\(\d{1,2}[\'\’]*\)'  # (1)
    subst_code_regex_5 = r'\([a-zABD-GI-Z]+\d+[\'\’]*\)'  # (a1)
    subst_code_regex_6 = r'\s[a-zABD-GI-Z]+\d+[\'\’]*[\s\.]'  #a1, ограниченный пробелами
    subst_code_regex_7 = r'\([a-zA-Z]{1,5}\-\d+[a-zABD-GI-Z]+[\'\’]*\)'  # (anti-1a)
    subst_code_regex_8 = r'\(\d+[a-zABD-GI-Z]+[\'\’]*\-\d+\)' #(2f-1)
    subst_code_regex_9 = r'\s\d+[a-zABD-GI-Z]+[\'\’]*\-\d+\s' # 2f-1 , ограниченный пробелами
    subst_code_regex_10 = r'\([a-zABD-GI-Z]{1,2}\d+[\'\’]*\-\d+\)' #(a1-1)
    subst_code_regex_11 = r'\s[a-zABD-GI-Z]{1,2}\d+[\'\’]*\-\d+\s' # a1-1 , ограниченный пробелами
    subst_code_regex_12 = r'\([A-Z]+\-?[A-Z]*\)' # (PD-HMF)
    
    code_list = re.findall(subst_code_regex_1, exp_sec_new)
    code_list += re.findall(subst_code_regex_2, exp_sec_new)
    code_list += re.findall(subst_code_regex_3, exp_sec_new)
    code_list += re.findall(subst_code_regex_4, exp_sec_new)
    code_list += re.findall(subst_code_regex_5, exp_sec_new)
    code_list += re.findall(subst_code_regex_6, exp_sec_new)
    code_list += re.findall(subst_code_regex_7, exp_sec_new)
    code_list += re.findall(subst_code_regex_8, exp_sec_new)
    code_list += re.findall(subst_code_regex_9, exp_sec_new)
    code_list += re.findall(subst_code_regex_10, exp_sec_new)
    code_list += re.findall(subst_code_regex_11, exp_sec_new)
    code_list += re.findall(subst_code_regex_12, exp_sec_new)
    code_list = list(set(code_list)) #это список всего, что может быть кодом
    
    #это список того, чего в коде быть не должно
    stop_list = ['1H', '19F', '13C', '11B', '31P', '15N', '77Se', '7Li', '27Al', '29Si', '17O', '3Н', '3He', '10B', '23Na', '25Mg', '33S', '35Cl', '37Cl', '39K', '43Ca', '51V', '59Co', '65Cu', '67Zn', '79Br', '81Br', '83Kr', '87Rb', '111Cd', '113Cd', '127I', '129Xe', '133Cs', '195Pt', '199Hg', '205Tl', '207Pb', '18O', 'oC', '°C', '32S', '|']
    
    
    code_list_new = [] #очищенный список кодов
    for code in code_list:
        check = True
        for stop in stop_list:
            if stop in code:
                check = False
        for stop in elements: 
            if stop == code:
                check = False
        if check: 
            code_list_new.append(code)
            
    code_dict = {} #словарь вида код как он встречается в тексте: код в универсальной обертке
    for code in code_list_new:
        code_dict.update({code: ' %CODE%%COD%' + code.replace(' ','').replace('(','').replace(')','').replace('.','')+'%COD%%CODE% '})
    
    exp_sec_new_smi_code = [] #заменить все коды соединений на них же в универсальной обертке + выделить в отдельный токен
    for sent in exp_sec_new_smi:
        if 'Compound_' in sent or 'MiscInch_' in sent:
            exp_sec_new_smi_code.append(_remove_extra_spaces(sent))
            continue
        for key in list(code_dict.keys()):
            sent = sent.replace(key, code_dict[key])
        sent_splitted = sent.split('%CODE%')
        for sent_spl in sent_splitted:
            exp_sec_new_smi_code.append(_remove_extra_spaces(sent_spl))

    
    
    
    exp_sec_dict = [] #[{'sentence': '', 'sent_type': '', 'dist_before': ''}, ...]
    
    dist = 0
    for sent in exp_sec_new_smi_code:
        temp_dict = {}
        if 'Compound_' in sent or 'MiscInch_' in sent:
            temp_dict.update({'sentence': sent})
            temp_dict.update({'sent_type': 'smiles'})
            temp_dict.update({'dist_before': dist})
            dist = 0
            exp_sec_dict.append(temp_dict)
        elif '%COD%' in sent :
            temp_dict.update({'sentence': sent})
            temp_dict.update({'sent_type': 'code'})
            temp_dict.update({'dist_before': dist})
            dist = 0
            exp_sec_dict.append(temp_dict)
        elif ';' in sent or '.' in sent or ',' in sent:
            dist += len(sent) + 10
        else:
            dist += len(sent)
            
    repr_with_code = []
        
    for idx, temp_dict in enumerate(exp_sec_dict):
        if idx == len(exp_sec_dict) - 1:
            break
        try: 
            if temp_dict['sent_type'] == 'smiles' and exp_sec_dict[idx+1]['sent_type'] == 'code' and exp_sec_dict[idx+1]['dist_before'] <= 3:
                trash = False                
                for el in elements:
                    if el == exp_sec_dict[idx+1]['sentence'].replace('%COD%',''):
                        trash = True 
                if not trash:
                    repr_with_code.append({'compound': temp_dict['sentence'], 
                                         'code': exp_sec_dict[idx+1]['sentence'].replace('%COD%','')})
        except Exception as e:
            print('_get_repr_to_code_dict', e)
            pass
    
    return repr_with_code

def create_code2compid(exp_sec, joint_sorted_dict, compid_descr, code_df):
    #this function gets exp_sec, replaces all chemical names with compid ("Compound_23").
    #Then it isolates all compound codes, joins them with data from OCR and gives a resulting dict code2comid
    #extra compounds from code_df (from OCR) are assigned to new compids 
    
    exp_sec_new = _replace_chem_tokens(exp_sec, joint_sorted_dict)
    
    comps_with_codes = _get_repr_to_code_dict(exp_sec_new)
    
    
    #if the text contains more than 2 fits compound_XXX - code with the same compound and different codes - 
    # this means that this compound is a general depiction, not a Compound.
    # Example: "starting with sulfonylcyclopropane S6"
    try:
        text_code_df = pd.DataFrame(comps_with_codes).value_counts().reset_index()
        repeated_compounds = text_code_df[text_code_df.duplicated(subset = 'compound', keep = False)].\
                             value_counts(subset = 'compound').reset_index()
        do_not_consider = list(repeated_compounds[repeated_compounds['count'] > 2]['compound'])
    except:
        do_not_consider = []
    
    comps_with_codes = [x for x in comps_with_codes if x['compound'] not in do_not_consider]
    #replace back compound ids from do_not_consider with theirs names
    compid2name = {v:k for k,v in joint_sorted_dict.items()}
    for compid in do_not_consider:
        exp_sec_new = exp_sec_new.replace('%COMP%' + compid + '%COMP%', compid2name[compid] )
        
    text_code_df = pd.DataFrame(comps_with_codes).value_counts().reset_index()
    
    #maximal compound id
    compound_ids = [x for x in compid_descr.keys() if 'Compound_' in x]
    compound_ids = [int(x.split('_')[1]) for x in compound_ids]
    if len(compound_ids) == 0:
        max_id = 0
    else:
        max_id = max(compound_ids)   
    
    #columns in code_df: 'smiles','text','dist_to_sub','file_path'
    if code_df.shape[0] != 0:
        code_df = code_df [code_df ['smiles'].apply(lambda x: '*' not in x)].copy() #alreasy done - duplicates
        code_df = code_df[code_df ['dist_to_sub'] < 600].copy() #alreasy done - duplicates
        code_df = code_df.value_counts(subset = ['text', 'smiles']).reset_index() 
        code_df = code_df[code_df['text'].apply(lambda x: '|' not in x)]
    else:
        code_df = pd.DataFrame(columns = ['text', 'smiles', 'count'])
    
    smiles2compid = {v:k for k,v in compid_descr.items()}
    smiles2mol = {x:indigo.loadMolecule(x) for x in smiles2compid}
    for key in smiles2mol:
        smiles2mol[key].aromatize()

    def _fit_smiles(smiles):
        try: 
            mol = indigo.loadMolecule(smiles)
            mol.aromatize()
            for smiles_ref in smiles2mol:
                if indigo.exactMatch(mol, smiles2mol[smiles_ref]):
                    return smiles2compid[smiles_ref]    
        except Exception as e:
            print('_fit_smiles', e, smiles)
            return None
    
    #add new compound ids
#     print(code_df)
    if code_df.shape[0] == 0:
        code_df = pd.DataFrame(columns = ['text', 'smiles', 'count']) 
        
    code_df['compound_id'] = code_df['smiles'].apply(_fit_smiles)
    mask = code_df['compound_id'].isna()
    code_df.loc[mask, 'compound_id'] = ['Compound_' + str(x + max_id) for x in 
                                        range(1, len(code_df.loc[mask])+1)]
    
    compid_descr.update({x['compound_id']:x['smiles'] for x in code_df.loc[mask].to_dict('records')})
    code_df.drop('smiles', axis = 1, inplace = True)
    code_df.rename(columns = {'text': 'code', 'compound_id': 'compound'}, inplace = True)   
    
    code_df['score'] = code_df['count']*0.5
    text_code_df['score'] = text_code_df['count']
    whole_codes = pd.concat([code_df, text_code_df])
    
    list_of_codes = []
    for row in whole_codes.iterrows():
        code = row[1]['code']
        compid = row[1]['compound']
        score = whole_codes.loc[(whole_codes['code'] == code) & 
                                (whole_codes['compound'] == compid)]['score'].sum()

        list_of_codes.append({'code': code, 'comp_id': compid, 'score': score})
    
    if len(list_of_codes) > 0:
        code_scores = pd.DataFrame(list_of_codes).sort_values(by = 'score', ascending = False)
        code_scores.drop_duplicates(inplace = True)

        code_scores.drop_duplicates(subset = 'code', inplace = True)
        code2compid = {x['code']: x['comp_id'] for x in code_scores.to_dict('records')}   
    else: 
        code2compid = {}
    
    return code2compid, compid_descr, exp_sec_new




def _replace_codes(exp_sec_new, joint_sorted_dict):
    #CURRENTLY NOT USED. MAY BE INTEGRATE replacement and duplicate cleaning to create_code2compid
    #коды соединений
    comp_to_code, exp_smi_code = _get_repr_to_code_dict(exp_sec_new)
    code_list = [v for k,v in comp_to_code.items()]
    code_to_comp ={v:k for k,v in comp_to_code.items()}
    
    #коду должно соответствовать соединение с самым длинным названием по ИЮПАК.
    duplicated_codes = []

    for item in code_list :
        if code_list.count(item) > 1 and item not in duplicated_codes:
            duplicated_codes.append(item)
    
    dupl_code_to_repr = {}
    for code in duplicated_codes:
        repr_list = []
        for k,v in comp_to_code.items():
            if v == code:
                repr_list.append(k)
        dupl_code_to_repr.update({code: repr_list})
    for code, reprs in dupl_code_to_repr.items():
        for k,v in joint_sorted_dict.items(): #идем по словарю. Он отсортирован по убыванию длины. Какое обозначение первым встретится - то и наше
            if v in reprs: 
                code_to_comp.update({code: v})
                break
    
    
    #замена всех кодов соединений при условии, что они обрамлены разумными границами и не являются чисто числовыми

    open_border_regex = r'[\s\/\(\[]'
    close_border_regex = r'[\s\.\,\;\:\/\]\)]'
    for code in code_to_comp.keys():
        if not code.isnumeric():
            regex_to_find = open_border_regex +code + close_border_regex
            tokens_to_sub = re.findall(regex_to_find, exp_sec_new)
            for token in tokens_to_sub:
                exp_sec_new = exp_sec_new.replace(token, token.replace(code, code_to_comp[code]))
    return exp_sec_new, code_to_comp

def _get_compound_fraction(sent):
    comp_len = 0
    
    comp_regex = r'Compound_\d+'
    misc_regex = r'MiscInch_\d+'
    solv_regex = r'Solvent_\d+'
    comps = re.findall(comp_regex, sent)
    miscs = re.findall(misc_regex, sent)
    solvs = re.findall(solv_regex, sent)
    
    if len(solvs) + len(comps) + len(miscs) > 4:
        for comp in comps:
            comp_len += len(comp)
        for misc in miscs:
            comp_len += len(misc)
        for solv in solvs:
            comp_len += len(solv)            
        
        return comp_len/len(sent.replace(' ', ''))
    else:
        return 0
    
def _additional_tokenization(protocol):
    #5 компаундов перечислено подряд, и они занимают >50% предложения по непробельным символам - после добавить sep 
    #Commercially available  Compound_7 ,  MiscInch_88 ,  MiscInch_372 ,  MiscInch_357 ,  MiscInch_452 , and  MiscInch_558  were used.
    from chemdataextractor.doc import Paragraph
    
    para = Paragraph(protocol)
    for sent in para.sentences:
        if _get_compound_fraction(sent.text) > 0.5:
            protocol = protocol.replace(sent.text, sent.text+'%SEP%')
        
    return protocol


def final_text_processing(exp_sec_new):

    #если идет что-то вида ... %YI%%SEP% 'something' %YI%%SEP%  %SPEC%, при это перед первым %YI%%SEP% нет 
    #%SPEC% и длина 'something' < 50
    #тогда надо первый %YI%%SEP% убрать

    exp_sec_new = exp_sec_new.replace('%COMP%', ' ').replace('%REF%', '%SEP%').\
                              replace('%GP%', '%SEP%').replace('%SP%','%SEP%')
    exp_sec_new = re.sub(r'(?:%SEP%\s*)+', '%SEP%', exp_sec_new)
    exp_sec_new = re.sub(r'(?:%SPEC%\s*)+', '%SPEC%', exp_sec_new)
    exp_sec_new = exp_sec_new.replace('% %', '%%')

    regex = r'.{70}%SEP%[^\%]{5,70}%YI%%SEP%[^\%]{0,50}%SPEC%'  
    # элементы, где выход оказался отделен от методики сепами, и за этим всем идут спектры
    for item in re.findall(regex, exp_sec_new):
        if '%SEP%' not in item[:70]:
            item_new = item.replace('%YI%', ' ').replace('%SEP%', ' ') + '%SEP%'
            exp_sec_new = exp_sec_new.replace(item, item_new)
    exp_sec_new = exp_sec_new.replace('%SPEC%', '%SEP%' + 'a '*100 + '%SEP%')  #чтобы потом не слились вместе
    exp_sec_new = exp_sec_new.replace('%SEP%', ' %SEP% ')

    #"smth. 1.1." - указываеь на то, что это новый раздел. 
    section_num_regex = r'[^\d\s]+\.\s+\d+\.\d+\.\s+'
    sec_nums = re.findall(section_num_regex, exp_sec_new)
    for sec_num in sec_nums:
        exp_sec_new = exp_sec_new.replace(sec_num, sec_num + '%SEP% ')

    exp_sec_new = exp_sec_new.replace(' Characterization of', ' %SEP% Characterization of')
    exp_sec_new = exp_sec_new.replace(' Copies of', ' %SEP% Copies of')
    exp_sec_new = exp_sec_new.replace(' Compounds characterization ', ' %SEP% Compounds characterization ')
    exp_sec_new = _additional_tokenization(exp_sec_new)

    splitted = exp_sec_new.replace('%SPEC%', '\n').replace('%SEP%', '\n').replace('%REF%', '\n').replace('%YI%', '\n').replace('%GP%', '\n').replace('%SP%', '\n').replace('%COMP%', ' ').split('\n')

    token_list = []
    for token in splitted:
        #' 5. Synthesis' - удалить 5ку
        regex = r'^\s*\d+\.\s*\w{3,6}'
        sec_nums = re.findall(regex, token)
        for sec_num in sec_nums:
            token = token.replace(sec_num, re.sub(r'\d+\.', '', sec_num))
        if len(token) > 0:
            token_list.append(token)
    token_list = [x for x in token_list if len(x) > 3]
    
    return token_list