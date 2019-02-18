
import torch
import torch.nn as nn
import torch.nn.functional as f
from flagrun_weights import *

def relu(x):
    return np.maximum(x, 0)
                            
class ExpertPolicy(nn.Module):
    "The policy for the expert demonstrator. Simple multi-layer perceptron policy, no internal state"
    def __init__(self, observation_space, action_space):
        super(ExpertPolicy, self).__init__()
        self.weights_dense1 = nn.Linear(observation_space.shape[0], 256) 
        self.weights_dense2 = nn.Linear(256, 128) 
        self.weights_dense_final = nn.Linear(128, action_space.shape[0]) 
        
        self.weights_dense1.weight = torch.nn.Parameter(torch.Tensor(weights_dense1_w.T))
        self.weights_dense1.bias = torch.nn.Parameter(torch.Tensor(weights_dense1_b))
        
        self.weights_dense2.weight = torch.nn.Parameter(torch.Tensor(weights_dense2_w.T))
        self.weights_dense2.bias = torch.nn.Parameter(torch.Tensor(weights_dense2_b))
        
        self.weights_dense_final.weight = torch.nn.Parameter(torch.Tensor(weights_final_w.T))
        self.weights_dense_final.bias = torch.nn.Parameter(torch.Tensor(weights_final_b))
                      

    def forward(self, x):
        x = f.relu(self.weights_dense1(x))
        x = f.relu(self.weights_dense2(x))
        x = self.weights_dense_final(x)
        return x

        