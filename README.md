# Applying the Forward-Forward Algorithm to the Event-Based Sensing
This is an implementation of the "Applying the Forward-Forward Algorithm to the Event-Based Sensing" paper. 

Our paper: Accepted to The 7th International Conference on Machine Learning and Machine Intelligence (MLMI 2024)

[Original Forward-Forward paper](https://arxiv.org/pdf/2212.13345.pdf) 

The code base developed on the Implementation of **Extending the Forward Forward Algorithm** <br>
[Extending the Forward Forward Algorithm](https://arxiv.org/pdf/2307.04205.pdf) <br>
[Implementation of Extending the Forward Forward Algorithm](https://github.com/Ads-cmu/ForwardForward) <br>

## How to Run

To execute the code, run main.py.

The code performs tasks as defined in config.yml.

## Pretrained Models and Configurations

[Link to pretrained models](https://drive.google.com/drive/folders/1WWv2uvlVXp3eaerJ0P0QrBhvDq46dFt-?usp=sharing) <be>

All the configurations YAMLs used in the paper are in the configs_models folder.

## Testing Models

For testing pretrained models create a folder in configs_models/(MNIST|NMNIST) and copy your yaml and model.

After that, you need to change paths in test.py and execute it.

You can see results in the folder that you create as a .txt file.

## How to Run on Google Colab

### Step 1: Clone the Repository and Set Up the Environment
To start, clone the repository using the following command in the terminal.
Set up the necessary libraries and login to Weights & Biases:
```bash 
!git clone https://github.com/SuleymanEmirAkin/Event-Based_Forward-Forward.git
!pip install omegaconf
!pip install hydra-core --upgrade
!pip install wandb
!pip install tonic
import wandb
wandb.login(key=YOUR_KEY)
```

### Step 2: Modify the Configuration
Adjust the configuration settings as necessary to fit your requirements.


### Step 3: Run the Script
Finally, execute the main script with:

```bash
!python Event-Based_Forward-Forward/main.py
```





