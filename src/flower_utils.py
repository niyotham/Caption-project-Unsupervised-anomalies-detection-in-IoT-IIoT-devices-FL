from copy import deepcopy
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy  as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from save_model import load_checkpoint_state, save_checkpoint_state
import warnings
from models import D_autoencoder, autoencoder,AE 
from  torch.optim.lr_scheduler import StepLR
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class IoTDataset(Dataset):

    def __init__(self, features, targets):
        self.n_samples =features.shape[0]
        self.features =features
        self.targets = targets
      
    def __getitem__(self, index):
        # dataset[0]s 
        return  self.features[index], self.targets[index]
    def __len__(self):
        #len(dataset) 
        return self.n_samples

def load_data(path,batch_size=32,testing_type: str=None):
    ''' Getting the original iot dataset.
    testing_type: str :-> either normal or anormal
    '''
    print('Loading IoT dataset....')
    # if path == None and train:
    #         raise ValueError('Please provide the path as a string for the train dataset')
    data = pd.read_csv(path)
    data= data.drop(['Attack_type'], axis=1)
    features = data.iloc[:,:-1]
    labels = data.iloc[:, -1]
    train_data, val_data, train_labels, val_labels = train_test_split(features, labels, 
                                                    test_size=0.2, random_state=42)
    sc= MinMaxScaler()
    x_train = sc.fit_transform(train_data.reset_index(drop=True))  
    x_test = sc.transform(val_data.reset_index(drop=True))
    trainset = IoTDataset(x_train, train_labels.reset_index(drop=True))

    if testing_type == None:
        # use the x_test for validating  the training phase
        testset = IoTDataset(x_test, val_labels.reset_index(drop=True))
    else:
        # upload the dataset for testing.
        path= 'dataset/DNN-EdgeIIoT_df_test.csv'  
        data = pd.read_csv(path)
        if testing_type=='normal':
            print('Getting normal dataset for testing')
            df_test_norm=data[data['Attack_label']==0]
            df_test_norm= df_test_norm.drop(['Attack_type'], axis=1)
            data_norm=df_test_norm.sample(frac=0.8).reset_index(drop=True)
            features = data_norm.iloc[:,:-1]
            labels = data_norm.iloc[:, -1]
            x_test = sc.transform(features)
            testset = IoTDataset(x_test, labels)
            
        elif testing_type== 'anormal':
            print('Getting anomaly dataset for testing')
            df_test_abnorm=data[data['Attack_label']==1]
            df_test_abnorm= df_test_abnorm.drop(['Attack_type'], axis=1)
            data_anorm=df_test_abnorm.sample(frac=0.8).reset_index(drop=True)
            features = data_anorm.iloc[:,:-1]
            labels = data_anorm.iloc[:, -1]
            x_test = sc.transform(features)
            testset = IoTDataset(x_test, labels)
            # testloader = DataLoader(testset, batch_size=batch_size,shuffle=True)
        else:
            raise Exception("Please provide testing_type argument as None,'normal' or 'abnormal'.")
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)  
    num_examples = {"trainset": len(trainset),'test_set':len(testset) }
    return trainloader,testloader,num_examples,trainset,testset


def train(model, train_loader, val_loader, n_epochs, resume=False,cid=None):
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler=StepLR(optimizer,step_size=1,gamma=0.9)
    criterion = nn.MSELoss(reduction='sum').to(DEVICE)
    history = dict(train=[], val=[], accuracy=[])
    if resume:
            model, start_epoch, optimizer,scheduler = load_checkpoint_state("checkpoint.pth.tar",
                                                        DEVICE, model,optimizer,scheduler)                                                                                                                                       
    start_epoch=1
    for epoch in range(start_epoch, n_epochs + 1):
        model = model.train()
        train_losses = []
        for feature, _ in train_loader:
            optimizer.zero_grad()
            feature_true = feature.float().to(DEVICE)
            feature_pred = model(feature_true.float())
            # get the reconstruction loss
            loss = criterion(feature_pred, feature_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        # Perfom validation on normal validation set
        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for feature,_ in val_loader:
                features = feature.to(DEVICE)
                pred_features = model(features.float())
                # get the validation reconstruction loss and append it to the val_losses
                recon_loss = criterion(pred_features, features)
                val_losses.append(recon_loss.item())
        # get the average loss
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        # append the loss to the history dict
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        # compute accuracy
        val_rec_loss=val_losses
        threshold=get_threshold(train_losses)
        norm_pred = sum(loss <= threshold for loss in val_rec_loss)
        acuracy_n= float(norm_pred/len(val_losses))
        history['accuracy'].append(acuracy_n)
        print(f'Epoch {epoch}: train loss {train_loss} validation loss {val_loss}')
    
    # compute the theshold of the whole training to help find the number of false positive
    rec_loss_train=history['train']
    threshold=get_threshold(rec_loss_train)
    val_rec_loss=history['val']
    # _,accuracy_norm,False_Pos=compute_metrics(threshold,rec_loss_train, val_loader, type='normal')
    # saving model  
    if cid !=None:
        path=f'data/partitions/{cid}/checkpoint1.pth.tar'
        torch.save(model.state_dict(), path) 
        torch.save(model.state_dict(), f'data/partitions/{cid}/model1.pth')  
    if cid ==None:
        path='checkpoint.pth.tar'
        torch.save(model.state_dict(), path) 
        torch.save(model.state_dict(), "model.pth") 
    save_checkpoint_state(epoch, model, optimizer, scheduler, history,threshold,cid_n=cid)
    _, recon_losses,_,_,accuracy,False_Pos=test(val_loader,test_type='normal',cid=cid)
    False_Pos_rate= False_Pos/len(val_loader)
    return history,recon_losses,threshold,False_Pos,False_Pos_rate,accuracy

def get_threshold(loss):
    threshold = np.mean(loss) + np.std(loss)
    # print("Threshold: ", threshold)
    return threshold

def test(testloader,test_type='abnormal',cid=None):
    criterion = nn.MSELoss(reduction='sum').to(DEVICE)
    model= D_autoencoder()
    if cid ==None:
        checkpoint_path='checkpoint.pth.tar'
        PATH='model.pth'
        model.load_state_dict(torch.load(PATH))
        # model.load_state_dict(checkpoint["model_state_dict"])
    if cid !=None:
        checkpoint_path= f'data/partitions/{cid}/checkpoint1.pth.tar'
        PATH=f'data/partitions/{cid}/model1.pth'
        # model.load_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(torch.load(PATH))
    # model.load_state_dict(checkpoint["model_state_dict"])
    print("Loading the checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    threshold= checkpoint['threshold']

    predictions, losses=[],[]
    with torch.no_grad():
        model.eval()
        for batch_idx, (feature,_ )in enumerate(testloader) :
            features= feature.float().to(DEVICE)
            pred_features= model(features.float())
            recon_loss= criterion(pred_features,features)
            losses.append(recon_loss.item())
            predictions.append(pred_features.cpu().numpy().flatten())
    n_features= len(testloader)

    # compute  pred_anorm,accuracy_anorm,False_Neg 
    prediction,accuracy,False_pred=compute_metrics(threshold,losses,testloader, type=test_type)
    return threshold, losses,n_features,prediction,accuracy,False_pred

def compute_metrics(threshold,recon_losses, dataloader,type=None):
    ''' type = 'normal' if the loss is for normal dataset otherwise must be set to abnormal 
       for anomaly dataset'''
    # computer accuracy, Presition, recall and  weighted F-measure   
    if type == None:
        raise Exception("Please provide the type of prediction as 'normal' or 'abnormal' ")
    if type == 'normal':
        norm_pred = int(sum(loss <= threshold for loss in recon_losses))
        print(f'Nomal predictions: {norm_pred}/{len(dataloader)}')
        accuracy_norm= float(norm_pred/len(dataloader))
        # 𝐹𝑃=𝑠𝑢𝑚(𝑦̂>𝑇𝑅) 
        False_Pos= sum(loss > threshold for loss in recon_losses)
        # False_Neg_rate= (n_features - pred_norm)/n_features
        return norm_pred,accuracy_norm,False_Pos
    if type == 'abnormal':
        anom_pred = int(sum(loss > threshold for loss in recon_losses))
        accuracy_anorm= float(anom_pred/len(dataloader))
        print(f'Abnomal predictions: {anom_pred}/{len(dataloader)}')
        False_Neg = sum(loss <= threshold for loss in recon_losses)
        return anom_pred,accuracy_anorm,False_Neg

def get_metrics(predictions,False_Pos,False_Neg):
    ''' computer accuracy, Presition, recall and F1_score'''
    # computer accuracy, Presition, recall and  weighted F-measure    
    precision = predictions / (predictions + False_Pos)
    recall = predictions / (predictions + False_Neg)
    f1_score = 2*((precision*recall)/(precision + recall))
    # plot_cm(pred_norm,pred_anorm,False_Pos,False_Neg)
    return precision, recall,f1_score
               
               
def plot_cm(TN,TP,FP,FN, name=None):
    ''' A function to plot a Confusion Matrix'''
    confusion_matrix = np.zeros((2,2 ))
    confusion_matrix[0][0]= float(TN* 100)
    confusion_matrix[0][1]= float(FP* 100)
    confusion_matrix[1][0]= float(FN* 100)
    confusion_matrix[1][1]= float(TP* 100)
    df_cm = pd.DataFrame(confusion_matrix, range(2), range(2))
    plt.figure(figsize=(10,7))
    ax=sns.heatmap(df_cm, vmin=df_cm.values.min(), vmax=1,fmt='.2f', square=True, cmap="YlGnBu",
        linewidths=0.1, annot=True, annot_kws={"fontsize":12,"size": 12}) 
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .2, .75, 1])
    cbar.set_ticklabels(['low', '20%', '75%', '100%'])
    if name !=None:
        plt.title(f'Device id_number {name}\'s confusion matrix',fontsize=14,fontstyle='italic')
        plt.savefig(f'devive_{name}_confusion_matrix.jpeg')
    if name ==None:
        plt.title(' Centralized confusion matrix',fontsize=14,fontstyle='italic')
        plt.savefig('Centralized confusion_matrix.jpeg')
    plt.ylabel('True Label',fontsize=14)
    plt.xlabel('Predicted Label',fontsize=14)
    plt.show()   
    
def plot_loss(train_history,name):
    ax = plt.figure().gca()
    ax.plot(train_history['train_loss'])
    ax.plot(train_history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'])
    plt.title('Loss over training epochs')
    plt.savefig(name)
    plt.show()

def main():
    print('Centralize pytorch training')
    print('Loading train data...')
    path = 'dataset/DNN-EdgeIIoT_train_normal.csv'
    trainloader, valloader,num_examples,_,_ =load_data(path,batch_size=32)
    print(len(trainloader.dataset))
    print(len(valloader.dataset))
   
    print('Crearing the model...')
    model=autoencoder().to(DEVICE).train()
    
    # with and L1 CRITERION with drop out no batch normalization with normal validatation
    print('Start training...')
    history = train(model, trainloader, valloader, n_epochs=50)
    # print(f'Train accuracy: {running_acc} | Train loss: {running_loss} | num_samples:{train_num_samples}')
    # print(f'Validation accuracy: {val_acc} | Validation loss: {val_loss} | num_samples:{len(valset)}')
    # print('evaluate the model on test set')
    # loss, accuracy,test_num_samples=test(net=net,testloader=testloader,step,device=DEVICE)
    # print(f'Test accuracy: {accuracy} | Test loss: {loss} | num_samples:{test_num_samples}')



if __name__=='__main__':
#    main()
#    load_data('dataset/DNN-EdgeIIoT_train_normal.csv', train=False)
    # model=autoencoder().to(DEVICE).train()  
    model=AE().to(DEVICE).train() 
    file_path='dataset/DNN-EdgeIIoT_train_normal.csv'
    trainloader,valloader,num_examples,trainset,testset=load_data(path=file_path)