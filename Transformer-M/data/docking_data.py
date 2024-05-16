from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
import pickle
import torch
import json

DS_Tasks = ['ADRB2', 'ABL1', 'CYT2C9', 'PPARG', 'GluA2', '3CL', 'HIVINT', 'HDAC2', 'KRAS', 'PDE5']
sub_tasks = ['docking_score', 'emodel_score', 'hbond_score', 'polar_score', 'coulombic_score']

class DockingDataset(InMemoryDataset):
    def __init__(self,
                 data_path='/home/AI4Science/fengsk/DockingData/dataset_v2.csv', task='ADRB2', split='train', pkl_path='/home/AI4Science/fengsk/DockingData/diversity_molecule_set.pkl', idx_data_path='/home/AI4Science/fengsk/DockingData/docking_id_idx_map.json'):
        
        all_data = pd.read_csv(data_path)
        split_lst = ['train', 'valid', 'test']
        assert split in split_lst
        sidx = split_lst.index(split)
        self.train_data = all_data[all_data['scaffold_folds'] == sidx]
        
        assert task in DS_Tasks
        this_task = [f'{task}_{st}' for st in sub_tasks]
        
        self.id_lst = self.train_data['IDs'].tolist()
        labels = []
        for tk in this_task:
            labels.append(self.train_data[tk].tolist())
        
        labels = np.array(labels)
        self.labels = np.transpose(labels)
        self.length = len(self.id_lst)
        
        with open(pkl_path, 'rb') as pa:
            self.raw_datas = pickle.load(pa)

        with open(idx_data_path, 'r') as fp:
            self.idx_map = json.load(fp)

            
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        id = self.id_lst[index]
        idx = self.idx_map[id]
        raw_data = self.raw_datas[idx]
        raw_data['target'] = self.labels[index]
        return raw_data
        # Process your sample here if needed
        rdkit_mol = AllChem.MolFromSmiles(smile)
        data = mol_to_graph_data_obj_simple(rdkit_mol)
        # manually add mol id
        data.id = index
        data.y = torch.tensor(self.labels[index], dtype=torch.float32)        
        data.y = data.y.reshape(1, -1)
        return data
