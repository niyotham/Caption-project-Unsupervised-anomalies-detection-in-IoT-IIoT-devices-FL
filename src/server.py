import flwr as fl
import sys
from pathlib import Path
from torch.utils.data import Subset, DataLoader
from typing import Dict, Optional, Tuple,List
from collections import OrderedDict
import argparse
import flwr as fl
from flwr.common import (Weights,weights_to_parameters,Parameters,Scalar,parameters_to_weights)
from flwr.common import Parameters
from flwr.common.typing import EvaluateIns, EvaluateRes,FitIns,FitRes
from flwr.server.client_manager import ClientManager, ClientProxy
import torch
from  flower_utils import load_data,test
# from utils import MLP, parseNumList
from models import D_autoencoder
import numpy as np
from flwr.server.strategy import FedAdam,FedAvg
import warnings
warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#   Write a custom Strategy which Inherits from Federated Averageing.

#   Custom Functionalities:
#       configure_evaluate: does also deliver a configure eval when a eval fn is selected
#                               --> does both SERVER SIDE and CLIENT SIDE eval

class Custom_Strategy(fl.server.strategy.FedAvg):
     # OVERRIDE configure_evaluate from FedAvg
    def configure_evaluate( self, rnd: int, parameters: Parameters, client_manager: ClientManager):
        """ Configure the next round of evaluation """
        # # DO CONFIGURE EVAL EVEN IF self.eval_fn is given
        # if self.eval_fn is not None:
            #  return []
        
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

stratey=Custom_Strategy()

# define a function that performs centralized evaluation
def centralized_eval(weights:Weights):
    ''' use the entire IoT test set to for centralized evaluation
    return  Optional[Tuple[float,Dict[str,Scalar]]]
    '''
    model= D_autoencoder().eval()
    # load the dataset
    path = str(Path('dataset/DNN-EdgeIIoT_train_normal.csv') )
    _, testloader,_,_,_=load_data(path,train=False)
    # update the parameters
    parameters=weights_to_parameters(weights)
    params_dict=zip(model.state_dict(), weights)
    state_dict=OrderedDict({k: torch.Tensor(v) for k,v in params_dict})
    model.load_state_dict(state_dict,strict=True)
    model.to(DEVICE)
    threshold, losses,n_features,pred_anorm,accuracy_anorm,False_Neg= test(path, model,testloader)

    metrics= {'centralized_acc':accuracy_anorm,'num_samples':n_features}
    print(metrics)

centralized_eval=None

def fit_config(rnd:int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        'opt_lr':0.001,
        "current_round": rnd,  # The current round of federated learning
        "local_epochs": 10,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    # val_steps = 5 if rnd < 4 else 5
    config={"batch_size": 32,}
    return config

def main():
    """
    # Load model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    """
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument('--cid', type=parseNumList,required=True, default=0-1)
    args = parser.parse_args()
    #path_to_data=path + "/"+ str(args.cid ) + "/" + "train.pt"

    model=D_autoencoder().to(DEVICE).train()
    model_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    strategy=FedAvg(
        fraction_fit=1.,
        fraction_eval=1.,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        eval_fn=None,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=weights_to_parameters(model_weights),
        )
    
   # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:8080", 
                         config={"num_rounds": 5},
                        grpc_max_message_length = int(536_870_912) ,
                        strategy=strategy)

if __name__ == "__main__":
    main()

