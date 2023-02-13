# Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL

This repository contains the code for the the caption project submitted to be able to graduate in masters of computing at The American University in Cairo.


## Introduction
![Alter text](https://www.google.com/imgres?imgurl=https%3A%2F%2Frepository-images.githubusercontent.com%2F241095326%2F04ad19b5-b049-4d07-8555-60699f80f0d8&imgrefurl=https%3A%2F%2Fgithub.com%2Fadap%2Fflower&tbnid=GA2Kl5WIUCVWNM&vet=12ahUKEwiKtoGT9ZL9AhWAticCHaclCNwQMygAegQIARA8..i&docid=_Ww1vof_1_-Z0M&w=1280&h=640&q=federated%20learning%20using%20flower%20git%20github&ved=2ahUKEwiKtoGT9ZL9AhWAticCHaclCNwQMygAegQIARA8)
Harnessing the potential of AI in  industiries leveraging IoT and IIoT devices requires access to huge amounts of data produced by them to develop robust models. One of the  solutions to prevent the problems of sharing sensitive data of IoT DEVICES  is to build secure  models by using distributer machine learning such as  federated learning. In this scenario, different models are trained on each device's local data and share their knowledge (parameters) with a central server that performs the aggregation in order to achieve a more robust and fair model. Note that in this project we used two devices and compared their results with 

This repository contains the code to reproduce the experiments performed in the framework of the Decentralized AI in Healthcare project at Sahlgrenska University Hospital and AI Sweden. In this example repository we are working on publicly available data (ISIC Archive) and simulating the decentralised setup internally. We have two different tasks on which we are actively working :




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

