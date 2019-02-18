import torch
import torch.nn as nn
import torch.nn.functional as f 
import time

from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as f


def rollout_for_one_episode(policy, env,  render=True):
    '''
    
    Rollout a particular policy for a single episode.
    
    '''
    
    rollout_data = {'observations':[], 'actions':[], 'score':[]}
    pi = policy
    
    frame = 0
    score = 0
    restart_delay = 0
    obs = env.reset()
    from itertools import count
    for t in count():
        rollout_data['observations'].append(obs)
        a = pi(torch.Tensor(obs)).data.numpy()
        import pdb
        rollout_data['actions'].append(a)
        obs, r, done, _ = env.step(a)
        score += r
        frame += 1
        if (render):
          time.sleep(1./60)

        still_open = env.render("human")

        if still_open==False:
            return
        if not done: continue
        if restart_delay==0:
            print("score=%0.2f in %i frames" % (score, frame))
            if still_open!=True:      # not True in multiplayer or non-Roboschool environment
                break
            restart_delay = 60*2  # 2 sec at 60 fps
        restart_delay -= 1
        if restart_delay==0: break
    rollout_data['score'].append(score)
    return rollout_data


def rollout_for_n_episodes(n, policy=None, env=None, render=True):
    '''
    Rollout a particular policy for a n episodes.
    '''
    
    if not policy: policy= ExpertPolicy(env.observation_space, env.action_space)
    rollout_data = {'observations':[], 'actions':[], 'scores':[]}
    for i in range(n):
        print('episode', i)
        recent_rollout_data = rollout_for_one_episode(policy, env, render=render)
        rollout_data['observations'].extend(recent_rollout_data['observations'])
        rollout_data['actions'].extend(recent_rollout_data['actions'])
        rollout_data['scores'].extend(recent_rollout_data['score'])
    return rollout_data


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y):
        'Initialization'
        self.X=X
        self.Y=Y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.X[index]
        Y = self.Y[index]
        return X, Y

def train_model(policy, training_data, num_epochs = 300):
    '''
    Given a dict of training data, train a policy network
    using supervised learning.
    
    '''
    
    dataset = Dataset(training_data['observations'], training_data['actions'])
    dataloader = data.DataLoader(dataset, batch_size = 128)
    
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for ne in range(num_epochs):
        for obs, act in dataloader:

            obs = Variable(obs)
            act = Variable(act)

            policy.zero_grad()
            optimizer.zero_grad()

            predicted_action = policy(obs)
            loss = mse(predicted_action, act.float())

            loss.backward(retain_graph=True)
            optimizer.step()

        if ne%50==0: print("Epoch: {}, Total loss: {}".format(ne, loss))
        
    return policy
            
