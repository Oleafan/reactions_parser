import re
import copy
from rdkit import Chem
from brutto_formulas import br_d
from dictionary import solv_smiles2name
from  indigo import Indigo
indigo = Indigo()
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

#usage: smiles = recognize_string(string) where string is something like 'Cy2PMe'
#ignores spaces, '-', '*'

def _join_molecules(mol1, atom_idx_1, mol2, atom_idx_2, bond_type):
    #bond_type: int
    #две молекулы. Соединять по R-сайтам. 
    combo = Chem.CombineMols(mol1,mol2)
    if bond_type == 0:
        return combo
    
    edcombo = Chem.EditableMol(combo)
    atom_1 = atom_idx_1
    atom_2 = atom_idx_2 + mol1.GetNumAtoms() #поскольку было объединение.
    
    #номера атомов для удаления.
    #для каждого атома собрать bond_type соседей, которые звездами обозначены
    dummies_to_remove_1 = []
    dummies_to_remove_2 = []
    
    atom = combo.GetAtomWithIdx(atom_1)
    for x in atom.GetNeighbors():
        if x.GetSymbol() == '*':
            dummies_to_remove_1.append(x.GetIdx())
        if len(dummies_to_remove_1) == bond_type:
            break
    
    atom = combo.GetAtomWithIdx(atom_2)
    for x in atom.GetNeighbors():
        if x.GetSymbol() == '*':
            dummies_to_remove_2.append(x.GetIdx())
        if len(dummies_to_remove_2) == bond_type:
            break
    
    if len(dummies_to_remove_1) == bond_type and len(dummies_to_remove_2) == bond_type:
        atoms_to_remove = dummies_to_remove_1 + dummies_to_remove_2
        atoms_to_remove.sort()
        if bond_type == 1:
            order = Chem.rdchem.BondType.SINGLE
        elif bond_type == 2:
            order = Chem.rdchem.BondType.DOUBLE
        elif bond_type == 3:
            order = Chem.rdchem.BondType.TRIPLE
        else:
            return None
        
        edcombo.AddBond(atom_1,atom_2,order=order)
        
        num_removed = 0
        for atom_idx in atoms_to_remove:
            edcombo.RemoveAtom(atom_idx - num_removed)
            num_removed += 1
        combo = edcombo.GetMol()
        return combo
    
    
def _num_r_groups(mol, atom_idx):
    r_groups = 0
    atom = mol.GetAtomWithIdx(atom_idx)
    for x in atom.GetNeighbors():
        if x.GetSymbol() == '*':
            r_groups += 1
    return r_groups

def _get_frag_list(splited, frag_dict):
    # нужно выстроить хитрую структуру данных, указывающую на связи между фрагами и их количество
    # idx - номер фрагмента
    # smiles - смайлс
    # num_repl - сколько раз повторяется
    # name - оставим сюда _FRAG\d_
    # mol - молекула
    # num_connect - число R-групп
    # {atom_no: num_r_groups}, номера атомов, при которых они находятся 
    # порядок - как в строке
    frag_list = []
    for idx, spl in enumerate(splited):
        temp_dict = {}
        temp_dict['idx'] = idx
        temp_dict['smiles'] = frag_dict[re.sub(r'_\d+', '_', spl)]
        temp_dict['num_repl'] = 1
        repls = re.findall(r'_\d+', spl)
        if len(repls) > 0:
            temp_dict['num_repl'] = int(repls[0].replace('_', ''))
        if temp_dict['num_repl'] > 40:
            return [] #чтобы избежать подвисания на лажовых примерах типа ['_RAG1_', '_RAG5_21511057', '_RAG1_', '_RAG6_2'] (возникло из 'H (2SB) 15110 57 H(2SC)')
        temp_dict['name'] = spl

        mol = Chem.MolFromSmiles(temp_dict['smiles'])
        temp_dict['mol'] = mol
        num_connect = 0
        connectors_dict = {}
        for atom in mol.GetAtoms(): 
            if not atom.GetSymbol() == '*':
                num_rs = _num_r_groups(mol, atom.GetIdx())
                if num_rs > 0:
                    num_connect += num_rs
                    connectors_dict[atom.GetIdx()] = num_rs
        temp_dict['num_connect'] = num_connect
        temp_dict['connectors_dict'] = connectors_dict
        frag_list.append(temp_dict)

    return frag_list

def _join_frags(frag_left, frag_right):
    #frag_left всегда имеет 1 num_replicates. frag_right - неограничено
    if len(frag_right['connectors_dict']) == 0:
        max_con_right = -1
    else:
        max_con_right = max(frag_right['connectors_dict'].values()) 
    if len(frag_left['connectors_dict']) == 0:
        max_con_left = -1
    else:
        max_con_left = max(frag_left['connectors_dict'].values()) 

    if max_con_left == -1 or max_con_right == -1:
        bond_type = 0
    else: 
        bond_type = min(max_con_left, max_con_right)
        
    for key in list(frag_right['connectors_dict'].keys()): #перебираем правый фрагмент с начала
        if frag_right['connectors_dict'][key] >= bond_type:
            atom_right = key
            break
    for key in list(frag_left['connectors_dict'].keys())[::-1]: #перебираем левый фрагмент фрагмент с конца
        if frag_left['connectors_dict'][key] >= bond_type:
            atom_left = key
            break
            
    mol = frag_left['mol']
    for idx in range(frag_right['num_repl']):
        #print(idx, Chem.MolToSmiles(mol), Chem.MolToSmiles(frag_right['mol']))
        if bond_type == 0:
            mol = _join_molecules(mol, 0 , frag_right['mol'], 0, bond_type) #atom_left, atom_right тут не используются по факту
            continue 
        
        #начинаем искать точку прикрепления к atom_left, так как она может сдвигаться из-за изменения числа атомов в молекуле
        #в начале уменьшаем вплоть до 2 атомов влево (мы удаляем один атом, именно на эту величину можно сместиться) 
        # а потом идем вправо на величину присоединяемой молекулы либо до окончания молекулы
        
        left_lim = max([0, atom_left-2])
        right_lim = min([atom_left + frag_right['mol'].GetNumAtoms(), mol.GetNumAtoms()])
        while atom_left >= left_lim and _num_r_groups(mol, atom_left) == 0:
            atom_left += -1
        if atom_left < 0:
            atom_left = 0
        while atom_left <= right_lim and _num_r_groups(mol, atom_left) == 0:
            atom_left += 1
        
        #print(atom_left, atom_right)
        #print('rgroups', _num_r_groups(mol, atom_left))
        new_mol = _join_molecules(mol, atom_left , frag_right['mol'], atom_right, bond_type)
        if new_mol is None and bond_type > 0:
            while new_mol is None or bond_type > 0:
                bond_type += -1
                new_mol = _join_molecules(mol, atom_left , frag_right['mol'], atom_right, bond_type)
        if new_mol is not None:
            mol = new_mol
    
    if mol is not None:
        result_mol = {}
        result_mol['mol'] = mol
        num_connect = 0
        connectors_dict = {}
        for atom in mol.GetAtoms(): 
            if not atom.GetSymbol() == '*':
                num_rs = _num_r_groups(mol, atom.GetIdx())
                if num_rs > 0:
                    num_connect += num_rs
                    connectors_dict[atom.GetIdx()] = num_rs
        result_mol['num_connect'] = num_connect
        result_mol['connectors_dict'] = connectors_dict
        return result_mol

def _build_molecule(frag_list):
    # надо этот фраглистпостепенно сворачивать в одну молекулу. Сейчас скобок нет. 
    # Идем слева направо. И коннкетим "жадно", то есть максимаьным количество связей. 
    # В левой группе для коннекта используем правые атомы, в правой группе - левые
    # На выходе получаем фрагмент в виде того же словаря, где есть 
    # 'mol', 'num_connect', 'connectors_dict'

    #выбрать стартовый фрагмент, от которого плясать. Ищем то, что повторяется один раз. 
    #От него сначала идем влево, а потом с тем, что получилось, вправо
    frag_list = copy.deepcopy(frag_list)
    for idx, frag in enumerate(frag_list):
        if frag['num_repl'] == 1:
            base_frag_idx = idx
            break

    base_frag = frag_list[base_frag_idx]
    #справа налево
    if base_frag_idx > 0:
        for frag in frag_list[base_frag_idx-1::-1]:
            #print(Chem.MolToSmiles(base_frag['mol']), Chem.MolToSmiles(frag['mol']))
            base_frag = _join_frags(base_frag, frag)
    
    #слева направо
    for frag in frag_list[base_frag_idx+1:]:
        #print(Chem.MolToSmiles(base_frag['mol']), Chem.MolToSmiles(frag['mol']))
        base_frag = _join_frags(base_frag, frag)
    return base_frag['mol']

def _build_mol_with_reverting(frag_list):
    try: 
        return _build_molecule(frag_list)
    except:
        reversed_frag_list = [x for x in frag_list[::-1]]
        return _build_molecule(reversed_frag_list)

def recognize_string(string):
    if len(string)>35:
        return None
    if len(string) < 2:
        return None
    if string.count(' ') > len(string)/10:
        return None
    if string.strip() == 'II' or string.strip() == '(II)':
        return None
    
    #попытка заменить по словарю dw.br_d
    for subst in br_d.keys():
        if subst.strip() == string.strip():
            return br_d[subst]
    string = string.replace('[', '(')
    string = string.replace(']', ')')
    string = string.replace(' ', '')
    string = string.replace('-', '')
    if '*' in string:
        string = string.replace('*', ')(')
        string = '('+ string + ')'
    
    #(2HF)
    aggl_mols = re.findall(r'\(\d+[^)(]+\)', string)
    for aggl in aggl_mols:
        num_repl = re.findall(r'\(\d+', aggl)[0]
        string = string.replace(aggl, aggl.replace(num_repl, '(')+num_repl.replace('(', ''))

    frag_dict = {}
    frag_count = 0
    for idx, key in enumerate(frags):
        if key in string:
            string = string.replace(key, '_RAG' + str(frag_count) + '_')
            frag_dict['_RAG' + str(frag_count) + '_'] = frags[key]
            frag_count += 1
    #находим скобку. преобразуем содержимое скобки в смайлс. Пополняем библиотеку фрагов
    frag_count = len(frag_dict) + 1
    
    if string.count('(') == string.count(')') and string.count('(') > 0:
        counter = 0
        while string.count('(') > 0:
            counter += 1
            if counter > 50:
                return None
            parath_list = re.findall(r'\([^)(]+\)', string)
            if len(parath_list) == 0:
                return None
            for parath in parath_list:
                
                splited = re.findall(r'_RAG\d+_\d*', parath)
                if len(splited) == 0:
                    return None
                if not '('+ ''.join(splited) + ')' == parath:
                    return None
                frag_list = _get_frag_list(splited, frag_dict)

                smiles = Chem.MolToSmiles(_build_mol_with_reverting(frag_list))
                string = string.replace(parath, '_RAG' + str(frag_count) + '_')
                frag_dict['_RAG' + str(frag_count) + '_'] = smiles
                frag_count += 1
    splited = re.findall(r'_RAG\d+_\d*', string)
    if not ''.join(splited) == string:
        return None
#     print(splited)
    frag_list = _get_frag_list(splited, frag_dict)
         
    mol = _build_mol_with_reverting(frag_list)  
    
    Chem.SanitizeMol(mol)
    smiles = Chem.MolToSmiles(mol)
    
    smiles = indigo.loadMolecule(smiles).smiles()
    if '*' in smiles:
        return None
    
    if len(smiles.split('.')) > 1:
        if len(set(smiles.split('.'))) ==  1:
            return None
    
    return smiles
    
#     edmol = Chem.EditableMol(mol)
#     atom_to_remove = []
#     for atom in mol.GetAtoms():
#         if atom.GetSymbol() == '*':
#             atom_to_remove.append(atom.GetIdx())
#     num_removed = 0
#     for atom_idx in atom_to_remove:
#         edmol.RemoveAtom(atom_idx - num_removed)
#         num_removed += 1
#     mol = edmol.GetMol()
#     Chem.SanitizeMol(mol)
#     smiles = Chem.MolToSmiles(mol)
    
#     return indigo.loadMolecule(smiles).smiles()



frags = {
        'Cy': 'C1CCCC(*)C1',
        'cyclohexyl': 'C1CCCC(*)C1',
        'Me': 'C*',
        'CH2': '*C([H])([H])*',
        'C': '*C(*)(*)*',
        'N': '*N(*)*',
        'P': '*P(*)*', #OS
        'H': '[H][*]',
        'O': '*O*',
        'He': '[He]',
        'Li': '[Li][*]',
        'Be': '[Be]([*])[*]',
        'B': '*B(*)*',
        'F': 'F*',
        'Na': '[Na][*]',
        'Mg': '[Mg]([*])[*]',
        'Al': '*[Al](*)*',
        'Si': '*[Si](*)(*)*', 
        'S': '*[S]*', #OS
        'Cl': 'Cl*', #OS
        'Ar': '[Ar]',
        'K': '[K][*]',
        'Ca': '[Ca]([*])[*]',
        'Sc':  '*[Sc](*)*', #OS
        'Ti': '*[Ti](*)*',#OS
        'V': '*[V](*)*',#OS
        'Cr': '[Cr]([*])[*]',#OS
        'Mn': '[Mn]([*])[*]',#OS
        'Fe': '[Fe]([*])[*]',#OS
        'Co': '[Co]([*])[*]',#OS
        'Ni': '[Ni]([*])[*]',#OS
        'Cu': '[Cu][*]',#OS
        'Zn': '[Zn]([*])[*]',
        'Ga': '*[Ga](*)*',#OS
        'Ge': '*[Ge](*)*',#OS
        'As': '*[As](*)*', #OS
        'Se': '*[Se]*', #OS
        'Br': 'Br*',
        'Rb': '[Rb][*]',
        'Sr': '[Sr]([*])[*]',
        'Y': '*[Y](*)*', #OS
        'Zr': '*[Zr](*)*', #OS
        'Nb': '*[Nb](*)*', #OS,
        'Mo': '*[Mo](*)(*)(*)*',#OS
        'Ru': '*[Ru](*)*', #OS,
        'Rh': '*[Rh](*)*', #OS,,
        'Pd': '[Pd]([*])[*]', #OS,, 
        'Ag': '[Ag][*]',#OS
        'Cd': '[Cd]([*])[*]',
        'In': '*[In](*)*', #OS,,,
        'Sn': '[Sn]([*])[*]', #OS,, 
        'Sb': '*[Sb](*)*', #OS
        'Te': '*[Te](*)(*)*',#OS
        'I': 'I*', #OS,
        'Xe': '*[Xe]*', #OS
        'Cs':  '[Cs][*]',
        'Ba': '[Ba]([*])[*]',
        'La': '*[La](*)*', #OS
        'Ce': '*[Ce](*)*', #OS,
        'Nd': '*[Nd](*)*', #OS,
        'Sm': '[Sm]([*])[*]',
        'Eu': '*[Eu](*)*', #OS,
        'Gd': '*[Gd](*)*', #OS,
        'Tb': '*[Tb](*)*', #OS,
        'Dy': '*[Dy](*)*', #OS,
        'Ho': '*[Ho](*)*', #OS,
        'Er': '*[Er](*)*', #OS,
        'Tm': '*[Tm](*)*', #OS,
        'Yb': '*[Yb](*)*', #OS,,
        'Lu': '*[Lu](*)*', #OS,,
        'Hf': '*[Hf](*)(*)*', #OS,,
        'Ta': '*[Ta](*)*', #OS,,
        'W': '*[W](*)*', #OS,,
        'Re': '*[Re](*)*', #OS,,
        'Os': '*[Os](*)*', #OS,,
        'Ir': '*[Ir](*)*', #OS,,
        'Pt': '*[Pt](*)(*)*', #OS,,
        'Au': '*[Au](*)*', #OS,,
        'Hg': '[Hg]([*])[*]',
        'Tl': '[Tl][*]',
        'Pb': '[Pb]([*])[*]',
        'Bi': '*[Bi](*)*', #OS,,
        'Ph':'*C1=CC=CC=C1',
        'SO2':'*S(=O)(=O)*',
        'SO3':'*S(=O)(=O)O*',
        'Et':'*CC',
        'Boc':'CC(C)(C)OC(*)=O',
        'Bz':'*C(=O)C1=CC=CC=C1',
        'Cbz':'*C(=O)OCC1=CC=CC=C1',
        'Ac':'CC(=O)*',
        'NO2':'[O-][N+](*)=O',
        'NO':'*N=O',
        'CN':'*C#N',
        'N3':'*N=[N+]=[N-]',
        'C2H5O':'*OCC',
        'C6H11':'*C1CCCCC1',
        'PMB':'COC1=CC=C(C*)C=C1',
        'PMP':'COC1=CC=C(*)C=C1',
        'Bn':'*CC1=CC=CC=C1',
        'Ms':'CS(*)(=O)=O',
        'Cys':'SC[C@H](N*)C(*)=O',
        'Pr':'*CCC',
        'Ts':'CC1=CC=C(C=C1)S(*)(=O)=O',
        'tBu':'CC(C)(C)*',
        't-Bu':'CC(C)(C)*',
        'But':'CC(C)(C)*',
        'Bu':'CCCC[*]',
        'nBu':'CCCC[*]',
        'n-Bu':'CCCC[*]',
        'Tf':'FC(F)(F)S([*])(=O)=O',
        'Tos':'CC1=CC=C(C=C1)S([*])(=O)=O',
        'FMOC':'*C(=O)OCC1C2=CC=CC=C2C2=C1C=CC=C2',
        'SO3-':'[O-]S(*)(=O)=O',
        'SO3H':'[OH]S(*)(=O)=O',
        'O-':'*[O-]',
        'NH3+':'*[NH3+]',
        'iPr':'CC(C)[*]',
        'i-Pr':'CC(C)[*]',
        'Pri':'CC(C)[*]',
        'CO3': 'C(=O)(O*)O*',
        'Bu3Sn': 'CCCC[Sn](*)(CCCC)CCCC',
        'nBu3Sn': 'CCCC[Sn](*)(CCCC)CCCC',
        'NO3': '[O-][N+](=O)O*',
        'SCN': 'C(S*)#N',
        'p-cymene': 'C1C=C(C(C)C)C=CC=1C',
        'p-Cymene': 'C1C=C(C(C)C)C=CC=1C',
        'SbF6': 'F[Sb](*)(F)(F)(F)(F)F',
        'O2C': 'O(*)C(*)=O',
        'OOC': 'O(*)C(*)=O',
        'C6F5': 'C1(=C(F)C(*)=C(F)C(F)=C1F)F',
        'MOM': 'COC*',
        'HMDS': 'C[Si](N(*)[Si](C)(C)C)(C)C',
        'acac': 'C(/C=C(/O*)\C)(=O)C',
        'bipy': 'N1=C(C=CC=C1)C1=NC=CC=C1',
        'bpy': 'N1=C(C=CC=C1)C1=NC=CC=C1',
        'phen': 'N1=CC=CC2=CC=C3C=CC=NC3=C12',
        'Sia': 'C(*)(C(C)C)C',
        'TMS': 'C[Si](*)(C)C',
        'Piv': 'C(*)(=O)C(C)(C)C',
        'TBS': '[Si](*)(C(C)(C)C)(C)C',
        'TBDMS': '[Si](*)(C(C)(C)C)(C)C',
        'TIPS': 'C([Si](*)(C(C)C)C(C)C)(C)C',
        'CO2': '*C(O*)=O',
        'pTol': '*C1C=CC(C)=CC=1',
        'p-Tol': '*C1C=CC(C)=CC=1',
        'oTol': '*C1C(C)=CC=CC=1',
        'o-Tol': '*C1C(C)=CC=CC=1',

        }

for smiles in solv_smiles2name:
    for solv_name in solv_smiles2name[smiles]:
        frags.update({solv_name: smiles})


frag_names = list(frags.keys())
frag_names.sort(key = len, reverse=True)
sorted_frags = {}
for key in frag_names:
    sorted_frags.update({key: frags[key]})
frags = sorted_frags