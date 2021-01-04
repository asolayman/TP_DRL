import copy
import gym
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.flatten_observation import FlattenObservation
import math
import matplotlib.pyplot as plt
import numpy as np
from random import sample, randint, random
import skimage.transform
import skimage.color
import sys
import torch
import vizdoomgym

from agent import DQNAgent




#################################################################################
### HYPER-PARAMETRES
# ENVIRONNEMENT
# Nombre d'épisodes
# Nombre max de step par épisode (Si l'environnement n'a pas déjà de limite)

# AGENT
# Taille du buffer d'épisode
# Valeur initiale de epsilon (Pour l'exploration)
# Valeur du decay de epsilon    

USED_SET = 'VizdoomCorridor-v0_Train'  # 'VizdoomBasic-v0_ReTrain' / 'VizdoomBasic-v0_Test'
PARAM_SET = {
    'CartPole-v1_Train' : {
        'ENV': 'CartPole-v1',
        'LAST_STEP': 500,
        'WEIGHT_FILE': None,
        'MONITOR': True,
        'USE_GPU': False,
        
        'TRAIN': True,
        'RENDER_TRAIN': True,
        'NB_TRAIN_EPISODE': 100,
        'NB_MAX_TRAIN_STEP': 10000,
        
        'TEST': False,
        'RENDER_TEST': True,
        'NB_TEST_EPISODE': 10,
        'NB_MAX_TEST_STEP': 10000,
        
        'LR': 1e-3,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 128,

        'UPDATE_METHOD': 'soft',
        'UPDATE_TARGET_EVERY': 40,
        'ALPHA': 0.01,

        'GAMMA': 0.999,
        
        'BUFFER_SIZE': 100000,

        'GREED_METHOD': 'epsilon',
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.,
        'EPSILON_DECAY': 0.995,
        'TAU': 1,
        
        'STACK_FRAME': False,
        'STACK_SIZE': 0
    },
    
    'VizdoomCorridor-v0_Train' : {
        'ENV': 'VizdoomCorridor-v0',
        'LAST_STEP': 2100,
        'WEIGHT_FILE': None,
        'MONITOR': True,
        'USE_GPU': True,
        
        'TRAIN': True,
        'RENDER_TRAIN': False,
        'NB_TRAIN_EPISODE': 401,
        'NB_MAX_TRAIN_STEP': 10000,
        
        'TEST': False,
        'RENDER_TEST': True,
        'NB_TEST_EPISODE': 10,
        'NB_MAX_TEST_STEP': 10000,
        
        'LR': 1e-3,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 64,

        'UPDATE_METHOD': 'soft',
        'UPDATE_TARGET_EVERY': 80,
        'ALPHA': 0.01,

        'GAMMA': 0.999,
        
        'BUFFER_SIZE': 100000,

        'GREED_METHOD': 'epsilon',
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY': 0.99977,
        'TAU': 1,
        
        'STACK_FRAME': True,
        'STACK_SIZE': 4
    },
    
    'VizdoomBasic-v0_Train' : {
        'ENV': 'VizdoomBasic-v0',
        'LAST_STEP': 300,
        'WEIGHT_FILE': None,
        'MONITOR': False,
        'USE_GPU': True,
        
        'TRAIN': True,
        'RENDER_TRAIN': False,
        'NB_TRAIN_EPISODE': 401,
        'NB_MAX_TRAIN_STEP': 10000,
        
        'TEST': False,
        'RENDER_TEST': True,
        'NB_TEST_EPISODE': 10,
        'NB_MAX_TEST_STEP': 10000,
        
        'LR': 1e-3,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 64,

        'UPDATE_METHOD': 'hard',
        'UPDATE_TARGET_EVERY': 80,
        'ALPHA': 0.01,

        'GAMMA': 0.999,
        
        'BUFFER_SIZE': 100000,

        'GREED_METHOD': 'epsilon',
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.,
        'EPSILON_DECAY': 0.9991,
        'TAU': 1,
        
        'STACK_FRAME': False,
        'STACK_SIZE': 0
    },
    
    'VizdoomBasic-v0_Test' : {
        'ENV': 'VizdoomBasic-v0',
        'LAST_STEP': 300,
        'WEIGHT_FILE': 'best_networks/VizdoomBasic-v0_400.pt',
        'MONITOR': True,
        'USE_GPU': False,
        
        'TRAIN': False,
        'RENDER_TRAIN': True,
        'NB_TRAIN_EPISODE': 100,
        'NB_MAX_TRAIN_STEP': 10000,
        
        'TEST': True,
        'RENDER_TEST': True,
        'NB_TEST_EPISODE': 100,
        'NB_MAX_TEST_STEP': 10000,
        
        'LR': 1e-3,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 64,

        'UPDATE_METHOD': 'soft',
        'UPDATE_TARGET_EVERY': 80,
        'ALPHA': 0.01,

        'GAMMA': 0.999,
        
        'BUFFER_SIZE': 100000,

        'GREED_METHOD': 'none',
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.,
        'EPSILON_DECAY': 0.99977, 
        'TAU': 1,
        
        'STACK_FRAME': False,
        'STACK_SIZE': 0
    }
}

def getParam(name):
    return PARAM_SET[USED_SET][name]

IS_CART = (getParam('ENV') == 'CartPole-v1')
#################################################################################        





class CartPoleDQNAgent(DQNAgent):
    pass

    
    
class VizDoomDQNAgent(DQNAgent):
    def _preprocess_state(self, state):
        return preprocess_vizdoom(state, [120, 160])

    def _build_model(self, out_features):
        in_features = 1
        if getParam('STACK_FRAME'):
            in_features = getParam('STACK_SIZE')
    
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 16, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=5, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(6528, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, out_features)
        )
        
def preprocess_vizdoom(state, resolution):
    if getParam('STACK_FRAME'):
        stack_frame = []
        
        for t in state:
            img = t[0]        
            # Changement de resolution
            img = skimage.transform.resize(img, resolution)
            # Passage en noir et blanc
            img = skimage.color.rgb2gray(img)
            # Passage en format utilisable par pytorch
            img = img.astype(np.float32)
            
            stack_frame.append(img)
            
        stacked_images = np.stack(stack_frame)
        return stacked_images
    else:
        img = state[0]        
        # Changement de resolution
        img = skimage.transform.resize(img, resolution)
        # Passage en noir et blanc
        img = skimage.color.rgb2gray(img)
        # Passage en format utilisable par pytorch
        img = img.astype(np.float32)
        img = img.reshape([1, resolution[0], resolution[1]])
        
        return img
        
        
        
        
        
def train(env, agent):
    total_steps = 0.
    episode_lengths = []
    episode_rewards = []

    reward, done = 0, False
    for ep in range(getParam('NB_TRAIN_EPISODE')):
        state = env.reset()
        
        if getParam('STACK_FRAME'):
            state = [state]*getParam('STACK_SIZE')

        sum_loss = 0.
        sum_step = 0.
        sum_reward = 0.
        for step in range(getParam('NB_MAX_TRAIN_STEP')):
            if getParam('RENDER_TRAIN'):
                env.render()
                
            action = agent.act(state, reward, done)
            next_state, reward, done, info = env.step(action)
            
            if getParam('STACK_FRAME'):
                next_state = state[1:] + [next_state]
            
            if step+1 != getParam('LAST_STEP'):
                agent.add_interaction(state, action, next_state, reward, done)
            
            state = next_state
            
            sum_loss += agent.train_step(total_steps)
            sum_step += 1.
            sum_reward += reward
            total_steps += 1
            
            if done:
                break
              
        episode_lengths.append(sum_step)
        episode_rewards.append(sum_reward)
                
        if ep%10 == 0:
            torch.save(agent.model.state_dict(), './network/' + getParam('ENV') + '_' + str(ep) + '.pt')
        
        print('Ep: ', ep, sep='', end='')
        print('  |  Sum reward: ', episode_rewards[-1], sep='', end='')
        print('  |  Avg loss: ', sum_loss/episode_lengths[-1], sep='', end='')
        print('  |  Total steps: ', total_steps, sep='', end='')
        print()
        
    plt.scatter(list(range(getParam('NB_TRAIN_EPISODE'))), episode_rewards)
    plt.show()
        
        
        
def test(env, agent):
    total_steps = 0.
    episode_lengths = []
    episode_rewards = []

    reward, done = 0, False
    for ep in range(getParam('NB_TEST_EPISODE')):
        state = env.reset()

        if getParam('STACK_FRAME'):
            state = [state]*getParam('STACK_SIZE')
        
        sum_step = 0.
        sum_reward = 0.
        for step in range(getParam('NB_MAX_TEST_STEP')):
            if getParam('RENDER_TEST'):
                env.render()
                
            action = agent.act(state, reward, done)
            next_state, reward, done, info = env.step(action)
            
            if getParam('STACK_FRAME'):
                next_state = state[1:] + [next_state]
            
            state = next_state
            
            sum_step += 1.
            sum_reward += reward
            total_steps += 1
            
            if done:
                break
              
        episode_lengths.append(sum_step)
        episode_rewards.append(sum_reward)
                        
        print('Ep: ', ep, sep='', end='')
        print('  |  Sum reward: ', episode_rewards[-1], sep='', end='')
        print('  |  Total steps: ', total_steps, sep='', end='')
        print()
        
    plt.scatter(list(range(getParam('NB_TEST_EPISODE'))), episode_rewards)
    plt.show()
        
        
        
        
        
if __name__ == '__main__':
    gym.logger.set_level(gym.logger.INFO)

    if IS_CART:
        # Creation de l'environnement
        env = gym.make(getParam('ENV'))
        if getParam('MONITOR'):
            env = gym.wrappers.Monitor(env, directory='./replay/cartpole', video_callable=lambda x: x%10 == 0, force=True)
        env.seed(0)
    
        # Creation de l'agent
        agent = CartPoleDQNAgent(
            env.action_space.n,
            getParam('LR'),
            getParam('BATCH_SIZE'),
            getParam('BUFFER_SIZE'),
            getParam('UPDATE_METHOD'),
            getParam('UPDATE_TARGET_EVERY'),
            getParam('ALPHA'),
            getParam('GAMMA'),
            getParam('GREED_METHOD'),
            getParam('EPSILON_START'),
            getParam('EPSILON_END'),
            getParam('EPSILON_DECAY'),
            getParam('TAU'),
            getParam('USE_GPU'),
            getParam('WEIGHT_FILE')
        )
    else:
        # Creation de l'environnement
        env = gym.make(getParam('ENV'), depth=True, labels=True, position=True, health=True)
        if getParam('MONITOR'):
            env = gym.wrappers.Monitor(env, directory='./replay/vizdoom', video_callable=lambda x: x%10 == 0, force=True)
        env.seed(0)
    
        agent = VizDoomDQNAgent(
            env.action_space.n,
            getParam('LR'),
            getParam('BATCH_SIZE'),
            getParam('BUFFER_SIZE'),
            getParam('UPDATE_METHOD'),
            getParam('UPDATE_TARGET_EVERY'),
            getParam('ALPHA'),
            getParam('GAMMA'),
            getParam('GREED_METHOD'),
            getParam('EPSILON_START'),
            getParam('EPSILON_END'),
            getParam('EPSILON_DECAY'),
            getParam('TAU'),
            getParam('USE_GPU'),
            getParam('WEIGHT_FILE')
        )
    
    if getParam('TRAIN'):
        train(env, agent)
        
    if getParam('TEST'):
        test(env, agent)
    
    env.close()










