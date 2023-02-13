import argparse
import shutil
from genericpath import exists
from os import mkdir
from turtle import st
import torch
# import torchvision 
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import WeightedRandomSampler,Subset, TensorDataset,ConcatDataset, DataLoader
from flwr.dataset.utils.common import create_lda_partitions
from pathlib import Path
from typing import Tuple, Dict
from pickle import TRUE
from torch import nn
from flower_utils  import load_data


DATA_ROOT= './data'
DATA_PARTITIONS ='./data/partitions'

def convert_pythorch_dataset_to_xy(dataset:TensorDataset):
    samples =[]
    targets =[]
    for feature, target in dataset:
        # samples.append(feature.numpy())
        samples.append(feature)
        targets.append(target)
    samples= np.array(samples)
    targets= np.array(targets, dtype=int )
    return (samples,targets)


def partition_iot_dataset(num_partions: int, concentration: float):
    ''' Getting the original iot dataset'''
    print('Loading normal IoT dataset for training and validation')
    file_path = 'dataset/DNN-EdgeIIoT_train_normal.csv'
    _,_,num_examples,trainset,_ =load_data(path=file_path,batch_size=32,testing_type=None)
    n_train=num_examples['trainset']
    n_val=num_examples['trainset']
    valset = Subset(trainset, range(n_train - n_val, n_train))
    print('Loading  IoT dataset containing anomalies  for testing purpose...')
    _,_,_,_,testset =load_data(path=file_path,batch_size=32,testing_type='anormal')
    # converting the dataset to xy = Tuple[np.array, np.array]
    print('converting to XY ...')
    iot_trainset_xy = convert_pythorch_dataset_to_xy(trainset)
    iot_valdset_xy= convert_pythorch_dataset_to_xy(valset)

    iot_testset_xy= convert_pythorch_dataset_to_xy(testset)
    #  partitioning trainset
    print('Partitioning trainset......')
    partions_trainset,dirichlet_dist_train= create_lda_partitions(dataset=iot_trainset_xy,
                                                            dirichlet_dist=None,
                                                            num_partitions=num_partions,
                                                            concentration= concentration,
                                                            accept_imbalanced=True )
    #  partitioning validation set
    print('creating validation Partitions using previous dirichlet distribution....')
    partions_valdset,_= create_lda_partitions(dataset=iot_valdset_xy,
                                                            dirichlet_dist=dirichlet_dist_train,
                                                            num_partitions=num_partions,
                                                            concentration= concentration,
                                                             accept_imbalanced=True )
    #  partitioning testset
    print('creating Test Partitions using previous dirichlet distribution....')
    partions_testset,_= create_lda_partitions(
                            dataset=iot_testset_xy,
                            dirichlet_dist=dirichlet_dist_train,
                            num_partitions=num_partions,
                            concentration= concentration,
                            accept_imbalanced=True )
    return partions_trainset,partions_valdset, partions_testset


def load_partitions(cid: int, data_dir= DATA_PARTITIONS):
    
    load_dir = Path(DATA_PARTITIONS) / str(cid)
    # train  
    trainset_xy =torch.load(load_dir/'train.pt')
    tensor_x = torch.Tensor(trainset_xy[0])
    tensor_y = torch.LongTensor(trainset_xy[1])
    trainset= TensorDataset(tensor_x,tensor_y) 
    trainloader= DataLoader(trainset, batch_size= 32,drop_last=True,shuffle=True)

    # validation
    valtset_xy =torch.load(load_dir/'validation.pt')
    tensor_x = torch.Tensor(valtset_xy[0])
    tensor_y = torch.LongTensor(valtset_xy[1])
    validset= TensorDataset(tensor_x,tensor_y)
    valloader= DataLoader(validset, batch_size= 32,drop_last=True)
    
    # test
    test_xy =torch.load(load_dir/'test.pt')
    tensor_x = torch.Tensor(test_xy[0])
    tensor_y = torch.LongTensor(test_xy[1])
    testset= TensorDataset(tensor_x,tensor_y)
    testloader= DataLoader(testset, batch_size= 32,drop_last=True)
    return trainloader, valloader,testloader

def main():
    ''' Generating LDA partitions for IoT_network datasets.'''
    parser= argparse.ArgumentParser(
        description='Generating LDA partitions for IoT_network datasets')
    parser.add_argument("--num_partitions", 
                       type=int,
                       default= 4, 
                       required=True, 
                       help='Number of partitions in which to split the dataset')
    parser.add_argument('--save root',
                        type=str,
                        default=DATA_PARTITIONS, 
                        help='Choose where to save partion.')
    parser.add_argument('--alpha', 
                        type=float, 
                        default=10.0, 
                        help='Choose concentration for LDA.')

    args = parser.parse_args()
    
    # creating partitions
    train_partition, val_partitions,test_partitions = partition_iot_dataset(num_partions=args.num_partitions, 
                                                     concentration=args.alpha)
    #  save train partitions
    for idx, part in enumerate(train_partition):
        save_dir = Path(DATA_PARTITIONS)/ str(idx)
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok = True)
        torch.save(part,save_dir/ 'train.pt')

    #  save validation  partitions
    for idx, part in enumerate(val_partitions):
        save_dir = Path(DATA_PARTITIONS) / str(idx)
        torch.save(part,save_dir/ 'validation.pt')
    #  save test  partitions
    for idx, part in enumerate(test_partitions):
        save_dir = Path(DATA_PARTITIONS) / str(idx)
        torch.save(part,save_dir/ 'test.pt')
        
if __name__== '__main__':
    main()
    # trainset,testset,num_examples =load_data()
    # convert_pythorch_dataset_to_xy(trainset)
    

