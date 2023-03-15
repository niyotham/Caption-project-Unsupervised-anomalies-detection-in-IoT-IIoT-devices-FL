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


### Flower Server 

For simple workloads we can start a Flower server and leave all the configuration possibilities at their default values. In a file named ```server.py```, import Flower and start the server:

```python
import flwr as fl

fl.server.start_server(config={"num_rounds": 3})
```
<!--- Add additional args ADVANCED-->
There are three ways to customize the way Flower orchestrates the learning process on the server side:
   * Use an existing strategy, for example, FedAvg
   * Customize an existing strategy with callback functions
   * Implement a novel strategy
   
The strategy abstraction enables implementation of fully custom strategies. A strategy is basically the federated learning algorithm that runs on the server. Strategies decide how to sample clients, how to configure clients for training, how to aggregate updates, and how to evaluate models. Flower provides a few built-in strategies which are based on the same API described below.

#### Built-in strategies

Flower comes with a number of popular federated learning strategies built-in. A built-in strategy can be instantiated as follows:

```python
import flwr as fl

strategy = fl.server.strategy.FedAvg()
fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)
```
This creates a strategy with all parameters left at their default values and passes it to the _start_server_ function. It is usually recommended to adjust a few parameters during instantiation. Existing strategies provide several ways to customize their behaviour. Callback functions allow strategies to call user-provided code during execution.

The server can pass new configuration values to the client each round by providing a function to _on_fit_config_fn_. The provided function will be called by the strategy and must return a dictionary of configuration key values pairs that will be sent to the client. It must return a dictionary of arbitraty configuration values _client.fit_ and _client.evaluate_ functions during each round of federated learning.

```python
import flwr as fl

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": str(0.001),
            "batch_size": str(32),
        }
        return config

    return fit_config

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.1,
    min_fit_clients=10,
    min_available_clients=80,
    on_fit_config_fn=get_on_fit_config_fn(),
)
fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)
```
The _on_fit_config_fn_ can be used to pass arbitrary configuration values from server to client, and potentially change these values each round, for example, to adjust the learning rate. The client will receive the dictionary returned by the _on_fit_config_fn_ in its own _client.fit()_ function.

Similar to _on_fit_config_fn_, there is also _on_evaluate_config_fn_ to customize the configuration sent to _client.evaluate()_.

Server-side evaluation can be enabled by passing an evaluation function to _eval_fn_.

#### Implement a novel strategy

All strategy implementation are derived from the abstract base class _flwr.server.strategy.Strategy_, both built-in implementations and third party implementations. This means that custom strategy implementations have the exact same capabilities at their disposal as built-in ones.

The strategy abstraction defines a few abstract methods that need to be implemented:

```python
class Strategy(ABC):

    @abstractmethod
    def configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

    @abstractmethod
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:

    @abstractmethod
    def configure_evaluate(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:

    @abstractmethod
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:

    @abstractmethod
    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
```
Creating a new strategy means implmenting a new class derived from the abstract base class _Strategy_ which provides implementations for the previously shown abstract methods:
```python
class SotaStrategy(Strategy):

    def configure_fit(self, rnd, weights, client_manager):
        # Your implementation here

    def aggregate_fit(self, rnd, results, failures):
        # Your implementation here

    def configure_evaluate(self, rnd, weights, client_manager):
        # Your implementation here

    def aggregate_evaluate(self, rnd, results, failures):
        # Your implementation here

    def evaluate(self, weights):
        # Your implementation here
```        
More on implementing strategies: [Implementing Strategies - flower.dev](https://flower.dev/docs/implementing-strategies.html) 

### Flower Client

The Flower server interacts with clients through an interface called ```Client```. When the server selects a particular client for training, it sends training instructions over the network. The client receives those instructions and calls one of the ```Client``` methods to run your code (i.e., to train the melanoma classification network).

Flower provides a convenience class called ```NumPyClient``` which makes it easier to implement the ```Client``` interface when your workload uses PyTorch. Implementing ```NumPyClient``` usually means defining the following methods (```set_parameters``` is optional though):

* **get_parameters**: 
    * return the model weight as a list of NumPy ndarrays

* **set_parameters (optional)**: 
    * update the local model weights with the parameters received from the server

* **fit**: 
    * set the local model weights
    * train the local model 
    * receive the updated local model weights

* **evaluate**:
    * test the local model

which can be implemented in the following way:

```python
class Client(fl.client.NumPyClient):
    """Flower client implementing melanoma classification using PyTorch."""

    def __init__(
        self,
        model: Net,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict, 
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_properties(self, config):
        return {} 
    
    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        self.model = train(self.model, self.trainloader, self.valloader, self.num_examples)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]: 
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, auc, accuracy, f1 = val(self.model, self.testloader)

        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy), "auc": float(auc), "f1": float(f1)}
                
```

We can now create an instance of our class Client and add one line to actually run this client:

```python
fl.client.start_numpy_client("[::]:8080", client=Client())
```
That’s it for the client. We only have to implement Client or NumPyClient and call fl.client.start_client() or fl.client.start_numpy_client(). The string "[::]:8080" tells the client which server to connect to. In our case we can run the server and the client on the same machine, therefore we use "[::]:8080". If we run a truly federated workload with the server and clients running on different machines, all that needs to change is the server_address we point the client at.

### Evaluation

There are two main approaches to evaluate models in federated learning systems: centralized (or server-side) evaluation and federated (or client-side) evaluation. 

#### Centralized Evaluation

All built-in strategies support centalized evaluation by providing an evaluation function during initialization. An evaluation function is any function that can take the current global model parameters as input and return evaluation results:

```python
def get_eval_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, _, _ = utils.load_data()

    n_train = len(trainset) 
    # Use the last 5k training examples as a validation set
    valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = utils.test(model, valset)
        return loss, {"accuracy": accuracy}

    return

# Load the model for server-side parameter evaluation
 model = utils.load_efficientnet(classes=10).to(device)

# Create strategy
strategy = fl.server.strategy.FedAvg(
    # ... other FedAvg arguments
    eval_fn=get_eval_fn(model),
)

# Start Flower server for four rounds of federated learning
fl.server.start_server("[::]:8080", strategy=strategy)
```

#### Custom Strategies 
The _Strategy_ abstraction provides a method called _evaluate_ that can direcly be used to evaluate the current global model parameters. The current server implementation calls _evaluate_ after parameter aggregation and before federated evaluation (see next section).

#### Federated Evaluation

Client-side evaluation happens in the ```Client.evaluate``` method and can be configured from the server side.  Built-in strategies support the following arguments:

  * ```fraction_eval```: a float defining the fraction of clients that will be selected for evaluation. If fraction_eval is set to 0.1 and 100 clients are connected to the server, then 10 will be randomly selected for evaluation. If fraction_eval is set to 0.0, federated evaluation will be disabled.
  * ``` min_eval_clients```: an int: the minimum number of clients to be selected for evaluation. If fraction_eval is set to 0.1, min_eval_clients is set to 20, and 100 clients are connected to the server, then 20 clients will be selected for evaluation.
  * ```min_available_clients```: an int that defines the minimum number of clients which need to be connected to the server before a round of federated evaluation can start. If fewer than min_available_clients are connected to the server, the server will wait until more clients are connected before it continues to sample clients for evaluation.
  * ```on_evaluate_config_fn```: a function that returns a configuration dictionary which will be sent to the selected clients. The function will be called during each round and provides a convenient way to customize client-side evaluation from the server side, for example, to configure the number of validation steps performed.

```python
def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}

# Create strategy
strategy = fl.server.strategy.FedAvg(
    # ... other FedAvg agruments
    fraction_eval=0.2,
    min_eval_clients=2,
    min_available_clients=10,
    on_evaluate_config_fn=evaluate_config,
)

# Start Flower server for four rounds of federated learning
fl.server.start_server("[::]:8080", strategy=strategy)
```

### Saving Progress

The Flower server does not prescribe a way to persist model updates or evaluation results. Flower does not (yet) automatically save model updates on the server-side. It’s on the roadmap to provide a built-in way of doing this.

#### Model Checkpointing

Model updates can be persisted on the server-side by customizing _Strategy_ methods. Implementing custom strategies is always an option, but for many cases it may be more convenient to simply customize an existing strategy. The following code example defines a new `SaveModelStrategy` which customized the existing built-in _FedAvg_ strategy. In particular, it customizes _aggregate_fit_ by calling _aggregate_fit_ in the base class (FedAvg). It then continues to save returned (aggregated) weights before it returns those aggregated weights to the caller (i.e., the server):

```python
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy(
    # (same arguments as FedAvg here)
)
fl.server.start_server(strategy=strategy)
```
#### Aggregate Custom Evaluation Results
The same Strategy-customization approach can be used to aggregate custom evaluation results coming from individual clients. Clients can return custom metrics to the server by returning a dictionary:
```python
class Client(fl.client.NumPyClient):

    def get_parameters(self):
        # ...

    def fit(self, parameters, config):
        # ...

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)

        # Return results, including the custom accuracy metric
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}
```
The server can then use a customized strategy to aggregate the metrics provided in these dictionaries:

```python
class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)

# Create strategy and run server
strategy = AggregateCustomMetricStrategy(
    # (same arguments as FedAvg here)
)
fl.server.start_server(strategy=strategy)
```

#### SSL-enabled Server and Client

Please, follow the following guide to learn how a SSL-enabled secure Flower server can be started and how a Flower client can establish a secure connection to it.

[`Guide: SSL-enabled Server and Client`](https://flower.dev/docs/ssl-enabled-connections.html)

## IOT Anomaly detection  

### Train the model in a federated setup

With both client and server ready, we can now run everything and see federated learning in action. FL systems usually have a server and multiple clients. We therefore have to start the server first:

[`server.py`](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/src/server.py):  

 ```
python server.py \
    --server_address=$SERVER_ADDRESS \
    --rounds=10 \
    --sample_fraction=1.0 \
    --min_num_clients=2
 
 ```
  We performed The model evaluation of both centralized and in a decentralized models. 

Once the server is running we can start the clients in different terminals. Open a new terminal per client and start the client:

[`client.py`](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/src/client.py) per terminal:
```
client.py --cid <the client ID number>

```



### Local centralized training

To train the model in a centralized way in case you want to make a comparison, you can run: 

[`flower_utils.py`](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/src/flower_utils.py)  
```
python flower_utils.py 

```

### Experiment Results.

> Testing results for both centralized and decentralized approaches were computed with the help of MSE reconstruction loss and the threshold of the training phase.

![Alter text](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/images/train_deoising_AE_128n_16b_Tahn_Dropout(0.2).png)
![Alter text-1](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/images/train_device0.png "Device0 =20%x") ![Alter text-2](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/images/train_device1.png "Device1 =20%x")

Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/images/train_device0.png)  |  ![](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/images/train_device1.png)

> Centralized deep auto encoder vs  FedAvg deep auto encoder  performance comparizon. 

* As it can be seen from the  gure, the accuracy for the federatedapproach for all experiments came at 99.8% where the average accuracy of the non-FL baseline reached 99.7%. Looking at the testing results, the di erence seems marginal for the simulation, and the number of false positive rates is better for synchronous FL with 0.16%. 
* The potential of the federated-based approach could be the e ect of training on decentralized data generated from various devices. It remains to be seen if the non-synchronous FL approach, which is our next activity, will also produce better results.


![Alter text](https://github.com/niyotham/Caption-project-Unsupervised-anomalies-detection-in-IoT-IIoT-devices-FL/blob/main/images/overall_metrics.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



### Flower Original Repository - A Friendly Federated Learning Framework

Please, refer to the original Flower Repository for more documentation and tutorials [Flower Original GitHub repository](https://github.com/adap/flower)

#### Documentation to read

[Flower Docs](https://flower.dev/docs):
* [Installation](https://flower.dev/docs/installation.html)
* [Quickstart (TensorFlow)](https://flower.dev/docs/quickstart_tensorflow.html)
* [Quickstart (PyTorch)](https://flower.dev/docs/quickstart_pytorch.html)
* [Quickstart (Hugging Face [code example])](https://flower.dev/docs/quickstart_huggingface.html)
* [Quickstart (PyTorch Lightning [code example])](https://flower.dev/docs/quickstart_pytorch_lightning.html)
* [Quickstart (MXNet)](https://flower.dev/docs/example-mxnet-walk-through.html)
* [Quickstart (scikit-learn [code example])](https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist)
* [Quickstart (TFLite on Android [code example])](https://github.com/adap/flower/tree/main/examples/android)

#### Flower Usage Examples

A number of examples show different usage scenarios of Flower (in combination
with popular machine learning frameworks such as PyTorch or TensorFlow). To run
an example, first install the necessary extras:

[Usage Examples Documentation](https://flower.dev/docs/examples.html)

Quickstart examples:

* [Quickstart (TensorFlow)](https://github.com/adap/flower/tree/main/examples/quickstart_tensorflow)
* [Quickstart (PyTorch)](https://github.com/adap/flower/tree/main/examples/quickstart_pytorch)
* [Quickstart (Hugging Face)](https://github.com/adap/flower/tree/main/examples/quickstart_huggingface)
* [Quickstart (PyTorch Lightning)](https://github.com/adap/flower/tree/main/examples/quickstart_pytorch_lightning)
* [Quickstart (MXNet)](https://github.com/adap/flower/tree/main/examples/quickstart_mxnet)
* [Quickstart (scikit-learn)](https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist)
* [Quickstart (TFLite on Android)](https://github.com/adap/flower/tree/main/examples/android)

Other [examples](https://github.com/adap/flower/tree/main/examples):

* [Raspberry Pi & Nvidia Jetson Tutorial](https://github.com/adap/flower/tree/main/examples/embedded_devices)
* [Android & TFLite](https://github.com/adap/flower/tree/main/examples/android)
* [PyTorch: From Centralized to Federated](https://github.com/adap/flower/tree/main/examples/pytorch_from_centralized_to_federated)
* [MXNet: From Centralized to Federated](https://github.com/adap/flower/tree/main/examples/mxnet_from_centralized_to_federated)
* [Advanced Flower with TensorFlow/Keras](https://github.com/adap/flower/tree/main/examples/advanced_tensorflow)
* [Single-Machine Simulation of Federated Learning Systems](https://github.com/adap/flower/tree/main/examples/simulation)



<!-- CONTACT -->
## Contributor

👤 **Niyomukiza Thamar**

- GitHub: [Niyomukiza Thamar](https://github.com/)
- LinkedIn: [Niyomukiza Thamar](https://www.linkedin.com/in/)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
 

## Show US your support

Give  a ⭐ if you like this project! 



