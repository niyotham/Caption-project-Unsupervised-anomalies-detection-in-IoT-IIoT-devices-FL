# Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL

This repository contains the code for the the caption project submitted to be able to graduate in masters of computing at The American University in Cairo.



![Alter text](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/images/FLower%20fl.png)
## Introduction
Harnessing the potential of AI in  industiries leveraging IoT and IIoT devices requires access to huge amounts of data produced by them to develop robust models. One of the  solutions to prevent the problems of sharing sensitive data of IoT DEVICES  is to build secure  models by using distributer machine learning such as  federated learning. In this scenario, different models are trained on each device's local data and share their knowledge (parameters) with a central server that performs the aggregation in order to achieve a more robust and fair model. Note that in this project we used two devices and compared their results with 

This repository contains the code for the implementation of detecting anomalies in IIoT devices network without accessing the data. In this example repository we are working on publicly available data [Edge\_IIoTset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) on [kaggle](https://www.kaggle.com/), simulating the decentralised setup internally and compare with centralized ML. We have two different tasks on which we are actively working :


## File structure

``` 
Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL|
├── LICENSE
├── README.md
├── data
│   └── partitions
│       ├── 0
│       │   ├── 10_epoches
│       │   │   ├── checkpoint1.pth.tar
│       │   │   └── model1.pth
│       │   ├── checkpoint1.pth.tar
│       │   ├── last
│       │   │   ├── checkpoint1.pth.tar
│       │   │   └── model1.pth
│       │   ├── model1.pth
│       │   ├── per!epoch_training
│       │   │   ├── checkpoint1.pth.tar
│       │   │   └── model1.pth
│       │   ├── test.pt.zip
│       │   ├── train.pt.zip
│       │   └── validation.pt.zip
│       └── 1
│           ├── checkpoint1.pth.tar
│           ├── last
│           │   ├── checkpoint1.pth.tar
│           │   └── model1.pth
│           ├── model1.pth
│           ├── per_1epoch_training
│           │   ├── checkpoint1.pth.tar
│           │   └── model1.pth
│           ├── test.pt.zip
│           ├── train.pt.zip
│           └── validation.pt.zip
├── dataset
│   ├── DNN-EdgeIIoT_df_test.csv.dvc
│   ├── DNN-EdgeIIoT_train_mix.csv.dvc
│   └── DNN-EdgeIIoT_train_normal.csv.dvc
├── images
│   ├── Centralized confusion_matrix.jpeg
│   ├── devive_0_confusion_matrix.jpeg
│   ├── devive_1_confusion_matrix.jpeg
│   ├── devive_One_confusion_matrix.jpeg
│   ├── devive_Zero_confusion_matrix.jpeg
│   ├── overall_metrics.jpeg
│   ├── overall_metrics.png
│   ├── train.png
│   ├── train_deoising_AE_128n_16b_Tahn_Dropout(0.2).png
│   ├── train_device0.png
│   ├── train_device1.png
│   ├── train_final.jpeg
│   └── train_loss.png
├── metric_data_centralized.txt
├── metrics.xlsx
├── notebooks
│   ├── preprocessing.ipynb
│   └── train.ipynb
├── requirements.txt
├── results
│   ├── 0_eval_results.json
│   ├── 0_results.json
│   ├── 1_eval_results.json
│   └── 1_results.json
├── src
│   ├── client.py
│   ├── create_lda_partitions.py
│   ├── flower_utils.py
│   ├── models.py
│   ├── save_model.py
│   └── server.py
└── train_128_autoencoder_final.csv
``` 
## Flower framework
Flower is a user-friendly framework designed for implementing the Federated Learning approach.

### Installation
Installing the Flower framework requires Python 3.6 or higher version.

To install its stable version found on PyPI:

``` 
  pip install flwr
```

To install its latest (though unstable) releases:

``` 
pip install flwr-nightly 
``` 
To install its latest version from GitHub
```
pip install git+https://github.com/adap/flower.git 
```
### Requirements
```
python3 -m pip install requirements.txt

```

## Federated learning pipeline
A federated learning system needs two parts

1. ***Server*** Server. The server hosts the aggregation logic and makes sure all the devices have the latest and updated model parameters.
2. ***Client*** Client. The clients (devices or silos - hospitals in our specific use case) have a local model running on the local data.

In our use case, we will be following the below steps.

1. We  build a IOT unsupervised anomaly detection based on ```auto-encoder``` using [Edge\_IIoTset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) dataset from [kaggle].
2. Then trained the model on the local data in each client device.  We have 2 locally running models ON 2 seperate DEVICES.
3. Once our model is trained and our model parameters are ready, we try to connect with the server.
4. The server then either accepts or rejects the invitation to connect based on some policy. It is simply  a First Come First Serve policy.
5. If the connection goes through, the client sends the model parameters to the server.
6. The server waits for all 2 model parameters and then aggregates them thus making use of all the data in all the models.
7. This can happen for as many rounds as we want to train the data.
8. Then the server sends the updates weight parameters back to the clients.
9. The client will now use the weights for image classification.
10. The client will now use the weights for anomaly detection.

![Alter text](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/images/FL%20true%202.png)





## Running the code

Running the code has two distinct parts i.e. starting up the server and initiating the clients. Each of these steps are explained below.

### Staring the Federated Server
First thing we need to do is to run the Federated Server. This can be done by either directly running the ```server.py``` file (with appropriate arguments, all possible arguments are discusses in next section) located under the ```src``` folder using:

``` 
python server.py \
    --server_address=$SERVER_ADDRESS \
    --rounds=10 \
    --sample_fraction=1.0 \
    --min_num_clients=2
```










