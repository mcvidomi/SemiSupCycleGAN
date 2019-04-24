import numpy as np
from torch.utils.data import Dataset
import pdb


class ToyDataset_paired(Dataset):
    def __init__(self, A,B):

        self.A = A
        self.B = B

    def __getitem__(self, index):
        item_A = self.A[index]
        item_B = self.B[index]

        A_y = 'foo'
        B_y = 'foo'
        return {'A': item_A, 'B': item_B, 'A_y': A_y, 'B_y': B_y}

    def __len__(self):
        return len(self.A)-1



class ToyDataset(Dataset):
    def __init__(self, A_unp, B_unp, A_p, B_p):

        self.A = np.vstack((A_unp,A_p))
        self.B = np.vstack((B_unp,B_p))
        self.idx_unp = len(A_unp)

    def __getitem__(self, index):
        item_A = self.A[index]
        item_B = self.B[index]

        if index >= self.idx_unp:
            item_P_A = index
            item_P_B = index
        else:
            item_P_A = 0
            item_P_B = 0

        A_y = 'foo'
        B_y = 'foo'
        return {'A': item_A, 'B': item_B, 'P_A': item_P_A,'P_B': item_P_B ,'A_y': A_y, 'B_y': B_y}

    def __len__(self):
        return len(self.A)-1
