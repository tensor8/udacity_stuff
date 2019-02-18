import torch
import torch.nn as nn
import torch.nn.functional as f 
import numpy as np
import random

                        
class StudentPolicy(nn.Module):
    "Simple multi-layer perceptron policy, no internal state"
    def __init__(self, observation_space, action_space):
        super(StudentPolicy, self).__init__()
        torch.manual_seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        
        self.weights_dense1 = nn.Linear(observation_space.shape[0], 256) 
        self.weights_dense2 = nn.Linear(256, 128) 
        self.weights_dense_final = nn.Linear(128, action_space.shape[0]) 

        self.weights_dense1.weight.data.normal_(0.0, 0.02)
        self.weights_dense2.weight.data.normal_(0.0, 0.02)
        torch.nn.init.xavier_uniform_(self.weights_dense_final.weight)
        
        self.weights_dense1.bias.data.fill_(0.01)
        self.weights_dense2.bias.data.fill_(0.01)
        self.weights_dense_final.bias.data.fill_(0.01)

    def forward(self, x):
        x = f.relu(self.weights_dense1(x))
        x = f.relu(self.weights_dense2(x))
        x = self.weights_dense_final(x)
        return x
