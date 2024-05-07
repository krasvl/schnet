import os
from urllib import request
import tarfile
import torch
from torch.utils.data import Dataset

def get_z(atom: str):
        z = { 'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        return z[atom]

class QM9Dataset(Dataset):
    def __init__(self, records_dir):
        self.records_dir = records_dir
        self.tar_path = os.path.join(self.records_dir, "gdb9_tar.tar.bz2")
        self.raw_path = os.path.join(self.records_dir, "gdb9_raw")

        if not os.path.exists(records_dir):
            self.__download_gdb9()
        self.files = os.listdir(self.raw_path)

    def __download_gdb9(self):
        url = "https://ndownloader.figshare.com/files/3195389"
        if not os.path.isdir(self.records_dir):
            os.makedirs(self.records_dir)
        print("Downloading QM9 dataset...")
        request.urlretrieve(url, self.tar_path)
        print("Extracting QM9 dataset...")
        with tarfile.open(self.tar_path) as tar:
            tar.extractall(self.raw_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        to_float = lambda s: float(s.replace('*^', 'e'))
        file_path = os.path.join(self.raw_path, self.files[idx])
        with open(file_path, 'r') as f:
            try:
                na = int(f.readline())
                t = torch.tensor(
                    [to_float(x) for x in f.readline().split('\t')[1:-1]], 
                    dtype=torch.float
                )
                props = [line.split('\t') for line in f.readlines()[:na]]
                r = torch.tensor(
                    [[to_float(x) for x in line[1:-1]] for line in props],
                    dtype=torch.float
                )
                z = torch.tensor(
                    [get_z(line[0]) for line in props], 
                    dtype=torch.int
                )
                return z, r, t
            except Exception as ex:
                raise Exception(f"Cant parse file {file_path}") from ex
            
class InMemoQM9Dataset(Dataset):
    def __init__(self, records_dir):
        self.records_dir = records_dir
        self.tar_path = os.path.join(self.records_dir, "gdb9_tar.tar.bz2")
        self.raw_path = os.path.join(self.records_dir, "gdb9_raw")

        if not os.path.exists(records_dir):
            self.__download_gdb9()
        self.files = os.listdir(self.raw_path)
        self.__load()

    def __download_gdb9(self):
        url = "https://ndownloader.figshare.com/files/3195389"
        if not os.path.isdir(self.records_dir):
            os.makedirs(self.records_dir)
        print("Downloading QM9 dataset...")
        request.urlretrieve(url, self.tar_path)
        print("Extracting QM9 dataset...")
        with tarfile.open(self.tar_path) as tar:
            tar.extractall(self.raw_path)

    def __parse_file(self, file):
        to_float = lambda s: float(s.replace('*^', 'e'))
        file_path = os.path.join(self.raw_path, file)
        with open(file_path, 'r') as f:
            try:
                na = int(f.readline())

                t = torch.tensor(
                    [to_float(x) for x in f.readline().split('\t')[1:-1]], 
                    dtype=torch.float
                )
                props = [line.split('\t') for line in f.readlines()[:na]]
                r = torch.tensor(
                    [[to_float(x) for x in line[1:-1]] for line in props], 
                    dtype=torch.float
                )
                z =  torch.tensor(
                    [get_z(line[0]) for line in props], 
                    dtype=torch.int
                )

                return z, r, t
            except Exception as ex:
                raise Exception(f"Cant parse file {file}") from ex

    def __load(self):
        print('Loading files...')
        self.items = []
        files = os.listdir(self.raw_path)

        for i, file in enumerate(files):
            if i % 10000 == 0:
                print(f'Files {(i//10000)*10000}/{len(files)}')
            z, r, t = self.__parse_file(file)
            self.items.append((z, r, t))

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)
    
def cust_collate(batch):
    return batch[0]
            
class BatchQM9Dataset(Dataset):
    def __init__(self, records_dir, batch_size):
        self.batch_size = batch_size
        self.records_dir = records_dir
        self.tar_path = os.path.join(self.records_dir, "gdb9_tar.tar.bz2")
        self.raw_path = os.path.join(self.records_dir, "gdb9_raw")

        if not os.path.exists(records_dir):
            self.__download_gdb9()
        self.files = os.listdir(self.raw_path)
        self.__load()

    def __download_gdb9(self):
        url = "https://ndownloader.figshare.com/files/3195389"
        if not os.path.isdir(self.records_dir):
            os.makedirs(self.records_dir)
        print("Downloading QM9 dataset...")
        request.urlretrieve(url, self.tar_path)
        print("Extracting QM9 dataset...")
        with tarfile.open(self.tar_path) as tar:
            tar.extractall(self.raw_path)

    def __parse_file(self, file):
        to_float = lambda s: float(s.replace('*^', 'e'))
        file_path = os.path.join(self.raw_path, file)
        with open(file_path, 'r') as f:
            try:
                na = int(f.readline())

                t = [to_float(x) for x in f.readline().split('\t')[1:-1]]
                props = [line.split('\t') for line in f.readlines()[:na]]
                r = [[to_float(x) for x in line[1:-1]] for line in props]
                z = [get_z(line[0]) for line in props]

                return z, r, t
            except Exception as ex:
                raise Exception(f"Cant parse file {file}") from ex

    def __load(self):
        print('Loading files...')
        self.batches = {i: [[],[],[]] for i in range(10**3)}
        self.items = []
        files = os.listdir(self.raw_path)

        for i, file in enumerate(files):
            if i % 10000 == 0:
                print(f'Files {(i//10000)*10000}/{len(files)}')
            z, r, t = self.__parse_file(file)
            size = len(z)
            batch = self.batches[size]
            batch[0].append(z)
            batch[1].append(r)
            batch[2].append(t)
            if len(batch[0]) == self.batch_size:
                z = torch.tensor(batch[0], dtype=torch.int)
                r = torch.tensor(batch[1], dtype=torch.float)
                t = torch.tensor(batch[2], dtype=torch.float)
                self.items.append((z, r, t))
                self.batches[size] = [[],[],[]]

        for size in self.batches:
            batch = self.batches[size]
            if len(batch[0]) > 0:
                z = torch.tensor(batch[0], dtype=torch.int)
                r = torch.tensor(batch[1], dtype=torch.float)
                t = torch.tensor(batch[2], dtype=torch.float)
                self.items.append((z, r, t))

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)
    
def cust_collate(batch):
    return batch[0]