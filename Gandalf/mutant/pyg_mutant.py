
import re
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from Gandalf.utils import parse_protein_name_from_filename
from torch_geometric.data.data import Data

try:
    from graphein.protein.resi_atoms import RESI_THREE_TO_1
except:
    print("can't import RESI_THREE_TO_1 from graphein.protein.resi_atoms")



def split_AAS(AAS, three_to_one:bool = True):
    """split AAS

    Args:
        AAS (str): VAL32LEU

    Returns:
        (origin, pos, mutant): (VAL, 32, LEU)
    """
    AAS_list = re.split(r"([0-9]+)", AAS)
    origin = AAS_list[0] if AAS_list[0] else None  
    pos = int(AAS_list[1]) if AAS_list[1] else None  # 1-base
    mutant = AAS_list[2] if AAS_list[2] else None

    if three_to_one:
        origin = RESI_THREE_TO_1[origin]
        mutant = RESI_THREE_TO_1[mutant]
    return origin, pos, mutant



def get_sequence_from_Data(x:Data, attr:str = "node_id", three_to_one:bool = False, return_chain:bool = True):
    """Generate sequence from pyg Data by "node_id" attr, note ``chain_dict`` is consist of chian_list.
       ``chain_dict`` looks like: {"A":[["VAL", 1], ["LEU", 2]], "B":[["MET", 1], ["LEU", 2]]}. Generally it should be a dict consists of {chain:chain_list} like {"A":chain_list}
       ``chain_list`` looks like: [["VAL", 1], ["LEU", 2]]. Generally it should be a List consists of [aa, pos] like [[aa_1, _1_based_pos], [aa_2, _1_based_pos]]
       ``seq_list`` or ``sequence_list`` looks like: ["MET", "VAL"]

    Args:
        x (torch_geometric.data.data.Data): pyg data of protein
        attr (str, optional): node_id saved attr. Defaults to "node_id".
        three_to_one (bool, optional): Three letter or single letter of amino acids. Defaults to False.
        return_chain (bool, optional): return level on chain if true, else will return {"A":List[str]} even if this protein have one more chains. Defaults to True.

    Raises:
        AttributeError: if "node_id" is not exists, you should specific this on attr

    Returns:
        Dict: {"A":[["VAL", 1], ["LEU", 2]], "B":[["MET", 1], ["LEU", 2]]} or if return chain is False :{"A":["MET", "VAL"]}
    """
    if hasattr(x, attr):
        aa_list = getattr(x, attr)
    else:
        raise AttributeError(f"{attr} is not the attr of {x.__class__}, please check or change the attr parameters")

    chain_dict:Dict[str, List[Union[str, int]]] = {}
    for aa in aa_list:
        chain, three_letter_aa, pos_one_based = aa.split(":")
        pos_one_based = int(pos_one_based)

        if not return_chain:
            chain = "A"

        if three_to_one:
            letter_aa = RESI_THREE_TO_1[three_letter_aa]
        else:
            letter_aa = three_letter_aa
            del three_letter_aa

        #  add chain 
        if chain not in chain_dict.keys():
            chain_dict[chain] = []
            chain_dict[chain].append([letter_aa, pos_one_based])
        else:
            chain_dict[chain].append([letter_aa, pos_one_based])
    
    return chain_dict


def get_seq_list_from_chain_dict(chain_dict:Dict[str, List[Union[str, int]]], chain_selection:Optional[str] = None):
    """Return selected chain's sequence or will return chain "A" no matter how many chains there are

    Args:
       ``chain_dict`` looks like: {"A":[["VAL", 1], ["LEU", 2]], "B":[["MET", 1], ["LEU", 2]]}. Generally it should be a dict consists of {chain:chain_list} like {"A":chain_list}
        chain_selection (Optional[str], optional): e.g. "A", then will return sequence of "A". Defaults to None.

    Returns:
        _type_: _description_
    """
    chains = list(chain_dict.keys())
    if chain_selection is None:
        if len(chains) == 1:
            chain_selection = "A"
        else:
            chain_selection = chains[0]
            
    return [aa for aa, pos in chain_dict[chain_selection]]

def mutantSequence_by_AAS(sequence_list:List[str], AAS:str, three_to_one:bool = False) -> List[str]:
    """change the sequence aa by AAS

    Args:
        sequence_list (List[str]): a list of amino acids
        AAS (str): part of HGVS description, like "MET1VAL" means at pos 1 (1-based) MET -> VAL  or "LEU24ILE" means at pos 24 LEU -> ILE
        three_to_one (bool, optional): Three letter or single letter of amino acids. Defaults to False.

    Raises:
        ValueError: If AAS's origin is not equal to sequence_list[pos-1] (0-based, pos-1)

    Returns:
        List[str]: sequence_list with mutation
    """

    #  get the mutant nsSNP sequence
    origin, pos, mutant = split_AAS(AAS = AAS, three_to_one = three_to_one)
    _0_base_pos = pos - 1  # 1-pos to 0-base 
    #  check sequence[pos] is as same as origin aa in AAS record
    if not sequence_list[_0_base_pos] == origin:
        raise ValueError(f"this mutant not happened on this protein as you can see: raw sequence {pos} is {sequence_list[_0_base_pos]} while mutant is {origin}")
    sequence_list[_0_base_pos] = mutant
    return sequence_list

def get_mutantMaskTensor_from_seqListAndAAS(seq_list:List[str], AAS:str):
    chain_tensor = torch.zeros(len(seq_list))
    if AAS is not None:  # None will return zeros tensor with shape (len(seq_list))
        origin, _1_based_pos, mutant = split_AAS(AAS = AAS)
        _0_based_pos = _1_based_pos - 1
        chain_tensor[_0_based_pos] = 1 
    
    return chain_tensor

def get_mutantMaskTensor_from_chain_dict(chain_dict:Dict[str, List[Union[str, int]]], SNP:Union[str, Dict[str, str]], chain_selection:str = None):
    """from chain_dict to generate mutationMaskArray in ``torch.Tensor``

    Args:
       ``chain_dict`` looks like: {"A":[["VAL", 1], ["LEU", 2]], "B":[["MET", 1], ["LEU", 2]]}. Generally it should be a dict consists of {chain:chain_list} like {"A":chain_list}
        SNP (Union[str, Dict[str, str]]): SNP is a single AAS at ``str`` or ``dict``, which is not sure, and will automatically deal like AAS or {"A":AAS1, "B":AAS2}, Recommend to be a dict even if a single chain mutation. AAS should be part of HGVS description, like "MET1VAL" means at pos 1 (1-based) MET -> VAL  or "LEU24ILE" means at pos 24 LEU -> ILE. 
        chain_selection (Optional[str], optional): e.g. "A", then will return sequence of "A". Defaults to None.

    Returns:
        List: [chain_A_tensor, chian_B_tensor]
    """
    if isinstance(SNP, str):
        AAS_dict = {"A": AAS_dict}
    else:
        AAS_dict = SNP
    
    if isinstance(chain_dict,Dict) and isinstance(AAS_dict, Dict):
        chain_tensors = []
        for chain_selection, chain_list in chain_dict.items():
            if chain_selection in AAS_dict.keys():
                AAS = AAS_dict[chain_selection]
                seq_list = [aa for aa, pos in chain_list]
                chain_tensors.append(get_mutantMaskTensor_from_seqListAndAAS(seq_list = seq_list, AAS = AAS))
            else:
                AAS = None
                chain_tensors.append(get_mutantMaskTensor_from_seqListAndAAS(seq_list = seq_list, AAS = AAS))
        return chain_tensors
    else:
        return TypeError(f"{chain_dict.__class__} is not ``Dict[str, List[Union[str, int]]]``")

def get_mutantMask_from_chain_dict(chain_dict:Dict[str, List[Union[str, int]]], SNP:Union[str, Dict[str, str]], chain_selection:str = None):
    """from chain_dict to generate mutationMaskArray in ``torch.Tensor``

    Args:
       ``chain_dict`` looks like: {"A":[["VAL", 1], ["LEU", 2]], "B":[["MET", 1], ["LEU", 2]]}. Generally it should be a dict consists of {chain:chain_list} like {"A":chain_list}
        SNP (Union[str, Dict[str, str]]): SNP is a single AAS at ``str`` or ``dict``, which is not sure, and will automatically deal like AAS or {"A":AAS1, "B":AAS2}, Recommend to be a dict even if a single chain mutation. AAS should be part of HGVS description, like "MET1VAL" means at pos 1 (1-based) MET -> VAL  or "LEU24ILE" means at pos 24 LEU -> ILE. 
        chain_selection (Optional[str], optional): e.g. "A", then will return sequence of "A". Defaults to None.

    Returns:
        List: [chain_A_tensor, chian_B_tensor]
    """
    if isinstance(SNP, str):
        AAS_dict = {"A": AAS_dict}
    else:
        AAS_dict = SNP
    
    if isinstance(chain_dict,Dict) and isinstance(AAS_dict, Dict):
        chain_mask_list = []
        for chain_selection, chain_list in chain_dict.items():
            if chain_selection in AAS_dict.keys():
                AAS = AAS_dict[chain_selection]
                origin, _1_based_pos, mutant = split_AAS(AAS = AAS)
                chain_mask_list.append(_1_based_pos -1)

            else:
                chain_mask_list.append(None)
        return chain_mask_list
    else:
        return TypeError(f"{chain_dict.__class__} is not ``Dict[str, List[Union[str, int]]]``")




def get_mutantSequenceDict_from_chain_dict(chain_dict:Dict[str, List[Union[str, int]]], SNP:Union[str, Dict[str, str]], three_to_one:bool = False) -> Dict[str, List[Union[str, int]]]:
    """get sequence_dict with mutation by AAS, if have one more chain in chain_dict or sequence_dict and SNP is {"A":AAS1, "B":AAS2}, it will mapped by key; if only one AAS, it will be automatically mapped to chain A or chain_selection

    Args:
        chain_dict (Dict[str, List[Union[str, int]]]): {"A":[["VAL", 0], ["LEU", 1]], "B":[["MET", 0], ["LEU", 1]]}
        SNP (Union[str, Dict[str, str]]): SNP is a single AAS at ``str`` or ``dict``, which is not sure, and will automatically deal like AAS or {"A":AAS1, "B":AAS2}, Recommend to be a dict even if a single chain mutation. AAS should be part of HGVS description, like "MET1VAL" means at pos 1 (1-based) MET -> VAL  or "LEU24ILE" means at pos 24 LEU -> ILE. 
        # chain_selection (str, optional): just us for one AAS and want to map it into selected chain(chain_selection). Defaults to None.
        three_to_one (bool, optional): Three letter or single letter of amino acids. Defaults to False.

    Raises:
        KeyError: chain id is not in chain_dict
        ValueError: SNP is not a str or Dict

    Returns:
        Dict[str, List[str | int]]: return chain_dict with mutation
    """
    if isinstance(SNP, str):
        AAS_dict = {"A": AAS_dict}
    else:
        AAS_dict = SNP

    chain_dict_tmp = deepcopy(chain_dict)  # make a copy of chain_dict

    for chain_selection, chain_list in chain_dict_tmp.items():

        if chain_selection not in AAS_dict.keys():
            chain_dict_tmp[chain_selection] = chain_list 
            continue 
        elif chain_selection in AAS_dict.keys():
        # sequence_list, pos_list = list(map(list, zip(*chain_list)))
            AAS = AAS_dict[chain_selection]
            sequence_list = [aa for aa, pos in chain_list]
            pos_list = [pos for aa, pos in chain_list]

            sequence_list = mutantSequence_by_AAS(sequence_list = sequence_list, AAS = AAS, three_to_one = three_to_one)
            chain_dict_tmp[chain_selection] = [[aa, pos] for aa, pos in zip(sequence_list, pos_list)]
    return chain_dict_tmp



def generate_nsSNP_pyg(x:Data,
    SNP:Union[str, Dict[str, str]],
    node_metadata_functions:Optional[Dict[str, Callable]] = None,
    graph_label:Optional[Union[int, Dict[str, int]]] = None,
    # graphein_config: ProteinGraphConfig = ProteinGraphConfig(),
    three_to_one:bool = False):
    """turn protein pyg into pyg with mutation change and add some node_metadata_function from Graphein

    Args:
        x (torch_geometric.data.data.Data): pyg of protein by graphein
        SNP (Union[str, Dict[str, str]]): AAS or {"A":AAS1, "B":AAS2}, AAS should be part of HGVS description, like "MET1VAL" means at pos 1 (1-based) MET -> VAL  or "LEU24ILE" means at pos 24 LEU -> ILE
        node_metadata_functions (Optional[Dict[str, Callable]], optional): Graphein function see more at Graphein. Defaults to None.
        three_to_one (bool, optional): Three letter or single letter of amino acids. Defaults to False.

    Returns:
        _type_: pyg with mutation
    """
    
    data = deepcopy(x)  #  copy of data 

    if isinstance(SNP, str):
        SNP = {"A": SNP}

    chain_dict:Dict[str, Tuple(str, int)] = get_sequence_from_Data(x, attr = "node_id", three_to_one = three_to_one)
    mutation_chain_dict = get_mutantSequenceDict_from_chain_dict(chain_dict=chain_dict, SNP=SNP, three_to_one=three_to_one)    

    data["mutation_chain_dict"] = mutation_chain_dict
    data["graph_name"] = [f"{parse_protein_name_from_filename(data.name[0])} " + "-".join([f"{chain_selection}:{AAS}" for chain_selection,  AAS in SNP.items()])]
    # data["mutation_masked_tensor"] = get_mutantMaskTensor_from_chain_dict(chain_dict=chain_dict, SNP=SNP)
    data["mutation_masked_0_based_pos"] = get_mutantMask_from_chain_dict(chain_dict=chain_dict, SNP=SNP)

    if graph_label:
        if isinstance(graph_label, str):
            data["graph_label"] = {"A":graph_label}
        elif isinstance(graph_label, dict):
            data["graph_label"] = {chain_selection:(graph_label[chain_selection] if chain_selection in graph_label else None) for chain_selection in chain_dict.keys()}


    data["node_id"] = [f'{chain}:{aa}:{pos}' for chain, chain_list in mutation_chain_dict.items() for aa, pos in chain_list ]

    #  add some attr Tensor by node_metadata_functions
    if node_metadata_functions:
        for key, func in node_metadata_functions.items():
            
            protein_embedding = []  #  chain level embedding order follow with chian_dict
            for chain_selection, chain_list in mutation_chain_dict.items():  # chain_level embedding
                chain_embedding = np.array([func(n=None, d = {"residue_name": aa}) for aa, pos in chain_list])  # note: {"residue_name":aa} is for compatible to graphein node_metadata_functions as it accept ``node`` and deal with node["residue_name"]
                protein_embedding.append(chain_embedding)
            data[key] = protein_embedding
    return data
