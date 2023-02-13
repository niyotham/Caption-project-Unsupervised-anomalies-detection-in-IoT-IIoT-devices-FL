from ast import Dict
from dataclasses import dataclass
from unicodedata import name
import warnings
import matplotlib.pyplot as plt
import flwr as fl
import numpy as np
import torch
from pathlib import Path
import json
from create_lda_partitions import load_partitions
from  torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
from torch.utils.data import  DataLoader, Subset
from models import autoencoder,AE,D_autoencoder
from flower_utils import (train, test ,get_threshold, get_metrics,
                          load_data, compute_metrics,plot_loss)
import argparse
import shutil
import seaborn as sns
sns.set()
warnings.filterwarnings("ignore")


DEVICE: str = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')


# FL_ROUND=0
train_loss_list=[]
eval_list=[]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,model,trainloader,validloader,params:Dict,
                     path: str,client_id:int):
        '''params:         dict with values for Clients setup, also used to log all training values
           client_id:      nr of client (int)
           client_type:    Type of client
        '''
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.params = params.copy()
        self.path= path 
        self.client_id = client_id
        self.name= f"{params['experiment_nr']}_{client_id:01}"
        # For Analytics Purposes, save those values in params --> for later inspections
        self.params['name'] = self.name
        # For Logging of progress of FL process
        self.params['metrics'] = []
        self.params['losses'] = []

    def get_parameters(self):
        """Get parameters of the local model."""
        # self.model.train()
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        # raise Exception("Not implemented (server-side parameter initialization)")

    def set_parameters(self, parameters):
        """ Set the model parameters on the local model that are received from the server 
            Loop over the list of model parameters received as of NumPy ndarray 
            (think list of neural network layers)"""
        # model= autoencoder().train().to(device=DEVICE)
        model= D_autoencoder().train().to(device=DEVICE)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Train parameters on the locally held training set.
        Set model parameters, train model, return updated model parameters
        """
        # Update local model parameters
        self.set_parameters(parameters)
        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        current_round:int = config["current_round"]
        # Use values provided by the config
        print(f"[Client{self.client_id} round {current_round}] fit, config: {config}")
        # trai the model
        print('Training the model in the fit function')
        history,rec_losses,threshold,False_Pos,False_Pos_rate,accuracy= train(self.model,self.trainloader,
                                                                self.validloader,n_epochs=epochs,cid=self.client_id)
        # _, rec_losses,_,pred_norm,accuracy,False_Pos= test(self.model,self.validloader,
                                                            # test_type='normal')
        # print(history)
      
        train_dict_res = {
            "dataset": len(self.trainloader),
            'n_samples': len(self.validloader),
            "fl_round": current_round,
            "train_loss": history['train'],
            "val_loss": history['val'],
            "norm_test_loss":rec_losses,
            "train_accuracy": float(accuracy),
            # "norm_pred":int(pred_norm),
            "False_Positive":int(False_Pos),
            "False_Positive_rate":float(False_Pos_rate),
            'threshold':float(threshold),
            'acc_traing_by_round':self.params['metrics']
        }
        train_loss_list.append(train_dict_res)
        self.params['losses'].append(history['train'])
        self.params['metrics'].append(accuracy)
        print(self.params['metrics'])
        return self.get_parameters(),len(self.trainloader),{'accuracy_norm': float(accuracy), }
         
    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        self.set_parameters(parameters)
        # global FL_ROUND
        # print(f"FL Round:{FL_ROUND}")
        # Get config values
        # steps: int = config["val_steps"]
        batch_size:int=config['batch_size']
        # load the dataset 
        # _,testloader,num_examples,_,_=load_data(self.path,batch_size=batch_size,testing_type='anormal')
        _, _,testloader=load_partitions(self.client_id, data_dir= './data/partitions')

        # Evaluate global model parameters on the local test data and return results
        print('Testing the model in evaluate function')
        _, _,_,pred_anorm,_,False_Neg= test(testloader,test_type= 'abnormal', cid=self.client_id )

        # First get the current round's information and then compute norm predicted metrics using compute_metrics function
        current_train_loss_list=train_loss_list[-1]
        rec_losses=current_train_loss_list['norm_test_loss']
        threshold=train_loss_list[0]['threshold']
        pred_norm,_,False_Pos= compute_metrics(threshold,rec_losses,self.validloader,type = 'normal')
        accuracy = (pred_norm + pred_anorm) / (pred_norm + pred_anorm + False_Neg +False_Pos)
        # get precision, recall,f_measure by function get_metrics
        precision, recall,f_measure=get_metrics(pred_anorm,False_Pos,False_Neg)

        results={"accuracy": float(accuracy),
                'precision':float(precision), 
                'recall':float(recall),
                'f_measure':float(f_measure),
                'predicted_norm':pred_norm,
                'pred_anorm':pred_anorm,
                'False_Pos':False_Pos,
                'False_Neg': int(False_Neg)}
        # FL_ROUND += 1
        eval_list.append(results)
        print(results)
        return float(threshold),len(self.validloader),{"accuracy": float(accuracy),
                'precision':float(precision), 
                'recall':float(recall),
                'f_measure':float(f_measure),}

def main():
    # Parse command line argument `partition`
    parser= argparse.ArgumentParser(description="Flower")
    parser.add_argument("--cid", type=int, required=True,help='Define the Client Id') 

    args = parser.parse_args()
    # Load PyTorch model
    model=D_autoencoder().to(DEVICE).train()
    # Load a subset of Iot dataset to simulate the local data partition
    # trainloader,valLoader,testloader,_=load_data(args.cid)
    params = {'experiment_nr': 1}
    path= 'dataset/DNN-EdgeIIoT_train_normal.csv'
    #trainloader,testloader,num_examples,trainset,testset=load_data(self.path,batch_size=batch_size,
                                                          #testing_type='anormal')
    trainloader, valloader,_=load_partitions(args.cid,data_dir='./data/partitions')
    client = FlowerClient(model, trainloader, valloader,params,path=path,client_id=args.cid)
    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",  
        client=client, 
        grpc_max_message_length = int(536_870_912) )

    
    # Save train and evaluation loss and accuracy in json file
    with open(f"results/{args.cid}_results.json", mode="w+") as train_file:
        json.dump(train_loss_list, train_file)
        
    with open(f"results/{args.cid}_eval_results.json", mode="w+") as eval_file:
        json.dump(eval_list, eval_file)
        
    print('Client DONE!')

if __name__ == "__main__":
    main()