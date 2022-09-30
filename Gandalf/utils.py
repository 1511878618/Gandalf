'''
Author: Tingfeng Xu xutingfeng@big.ac.cn
Date: 2022-09-27 12:33:15
LastEditors: Tingfeng Xu xutingfeng@big.ac.cn
LastEditTime: 2022-09-30 09:00:10
FilePath: /deeplearning/GNN/Gandalf/utils.py
Description: 

Copyright (c) 2022 by Tingfeng Xu xutingfeng@big.ac.cn, All Rights Reserved. 
'''



import re
import os.path as osp
ALPHAFOLD_PATTERN = r"(?<=AF-)[^-]*(?=-)"

def get_uacc_from_af2(af2_Filename):
    return re.search(ALPHAFOLD_PATTERN, af2_Filename).group()

def parse_protein_name_from_filename(filePath):
    root_path, fileName = osp.splitext(filePath)
    fileName, file_suffix = osp.split(fileName)
    
    if re.search(ALPHAFOLD_PATTERN, fileName):
        return get_uacc_from_af2(fileName)
    else:
        return fileName
        

