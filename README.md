``` Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/
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
