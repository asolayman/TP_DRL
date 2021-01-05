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
# ENV                   Nom de l'environnement
# RESOLUTION            Résolution des images [Height, Width] (Utile qu'avec VizDoomGym)
# LAST_STEP             Nombre d'étapes max de l'environnement (Mettre 100000 si non connu)
# WEIGHT_FILE           Path du fichier de poids (None si auccun poids à fournir)
# MONITOR               Si True l'execution est enregistrée tout les 10 ep, sinon non
# USE_GPU               Si True utilise le GPU pour les calculs
# 
# TRAIN                 Si True passe par un entrainement
# RENDER_TRAIN          Si True l'environnement est render pendant l'entrainement
# NB_TRAIN_EPISODE      Nombre d'episodes à faire pendant l'entrainement
# NB_MAX_TRAIN_STEP     Nombre max d'étapes à faire pendant l'entrainement par épisodes (Mettre 100000 si non connu)
# 
# TEST                  Si True passe par phase de test
# RENDER_TEST           Si True l'environnement est render pendant la phase de test
# NB_TEST_EPISODE       Nombre d'episodes à faire pendant la phase de test
# NB_MAX_TEST_STEP      Nombre max d'étapes à faire pendant la phase de test par épisodes (Mettre 100000 si non connu)
# 
# LR                    Taux d'apprentissage
# WEIGHT_DECAY          Valeur du weight decay    
# BATCH_SIZE            Taille du batch
# 
# UPDATE_METHOD         Méthode d'update du dupliquata 'soft' = méthode avec alpha, 'hard' = deep copy
# UPDATE_TARGET_EVERY   Si UPDATE_METHOD=='hard', le nombre d'itération avant la deep copy
# ALPHA                 Si UPDATE_METHOD=='soft', la valeur d'alpha
# 
# GAMMA                 La valeur de gamma
# 
# BUFFER_SIZE           La taille de la mémoire de l'agent
# 
# GREED_METHOD          Méthode d'exploration, 'epsilon', 'boltzmann', ou 'none'
# EPSILON_START         Si GREED_METHOD=='epsilon', valeur de epsilon au début
# EPSILON_END           Si GREED_METHOD=='epsilon', valeur minimum de epsilon
# EPSILON_DECAY         Si GREED_METHOD=='epsilon', valeur du decay
# TAU                   Si GREED_METHOD=='boltzmann', valeur de tau
#
# STACK_FRAME           Si True utilise le frame stacking sinon non
# STACK_SIZE            Si STACK_FRAME==True, nombre de framse à stack



## Possible values for USED_SET
# 'CartPole-v1_Train'
# 'CartPole-v1_Test'
# 'VizdoomBasic-v0_Train'
# 'VizdoomBasic-v0_Test'
# 'VizdoomTakeCover-v0_Train'
# 'VizdoomTakeCover-v0_Test'
# 'VizdoomCorridor-v0_Kill'
# 'VizdoomCorridor-v0_Train'
# 'VizdoomCorridor-v0_Test'
USED_SET = 'VizdoomCorridor-v0_Test'
PARAM_SET = {
    'CartPole-v1_Train' : {
        'ENV': 'CartPole-v1',
        'LAST_STEP': 500,
        'WEIGHT_FILE': None,
        'MONITOR': True,
        'USE_GPU': False,
        
        'TRAIN': True,
        'RENDER_TRAIN': True,
        'NB_TRAIN_EPISODE': 300,
        'NB_MAX_TRAIN_STEP': 10000,
        
        'TEST': False,
        'RENDER_TEST': True,
        'NB_TEST_EPISODE': 10,
        'NB_MAX_TEST_STEP': 10000,
        
        'LR': 1e-3,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 128,

        'UPDATE_METHOD': 'hard',
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
    
    'CartPole-v1_Test' : {
        'ENV': 'CartPole-v1',
        'LAST_STEP': 500,
        'WEIGHT_FILE': 'best_networks/CartPole-v1_Best.pt',
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
        'BATCH_SIZE': 128,

        'UPDATE_METHOD': 'hard',
        'UPDATE_TARGET_EVERY': 40,
        'ALPHA': 0.01,

        'GAMMA': 0.999,
        
        'BUFFER_SIZE': 100000,

        'GREED_METHOD': 'none',
        'EPSILON_START': 0.05,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY': 1.0,
        'TAU': 1,
        
        'STACK_FRAME': False,
        'STACK_SIZE': 0
    },
    
    'VizdoomTakeCover-v0_Train' : {
        'ENV': 'VizdoomTakeCover-v0',
        'RESOLUTION': [60, 80],
        'LAST_STEP': 100000,
        'WEIGHT_FILE': None,
        'MONITOR': True,
        'USE_GPU': True,
        
        'TRAIN': True,
        'RENDER_TRAIN': False,
        'NB_TRAIN_EPISODE': 201,
        'NB_MAX_TRAIN_STEP': 10000,
        
        'TEST': False,
        'RENDER_TEST': True,
        'NB_TEST_EPISODE': 10,
        'NB_MAX_TEST_STEP': 10000,
        
        'LR': 1e-5,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 16,

        'UPDATE_METHOD': 'soft',
        'UPDATE_TARGET_EVERY': 80,
        'ALPHA': 0.01,

        'GAMMA': 0.999,
        
        'BUFFER_SIZE': 10000,

        'GREED_METHOD': 'epsilon',
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY': 0.9997,
        'TAU': 1,
        
        'STACK_FRAME': True,
        'STACK_SIZE': 4
    },
    
    'VizdoomTakeCover-v0_Test' : {
        'ENV': 'VizdoomTakeCover-v0',
        'RESOLUTION': [60, 80],
        'LAST_STEP': 100000,
        'WEIGHT_FILE': 'best_networks/VizdoomTakeCover-v0_Best.pt',
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
        
        'LR': 1e-4,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 16,

        'UPDATE_METHOD': 'soft',
        'UPDATE_TARGET_EVERY': 80,
        'ALPHA': 0.01,

        'GAMMA': 0.999,
        
        'BUFFER_SIZE': 10000,

        'GREED_METHOD': 'epsilon',
        'EPSILON_START': 0.05,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY': 1.0,
        'TAU': 1,
        
        'STACK_FRAME': True,
        'STACK_SIZE': 4
    },
    
    'VizdoomCorridor-v0_Train' : {
        'ENV': 'VizdoomCorridor-v0',
        'RESOLUTION': [60, 80],
        'LAST_STEP': 2100,
        'WEIGHT_FILE': None,
        'MONITOR': True,
        'USE_GPU': True,
        
        'TRAIN': True,
        'RENDER_TRAIN': False,
        'NB_TRAIN_EPISODE': 201,
        'NB_MAX_TRAIN_STEP': 10000,
        
        'TEST': False,
        'RENDER_TEST': True,
        'NB_TEST_EPISODE': 10,
        'NB_MAX_TEST_STEP': 10000,
        
        'LR': 1e-5,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 16,

        'UPDATE_METHOD': 'soft',
        'UPDATE_TARGET_EVERY': 80,
        'ALPHA': 0.01,

        'GAMMA': 0.1,
        
        'BUFFER_SIZE': 10000,

        'GREED_METHOD': 'epsilon',
        'EPSILON_START': 1.0,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY': 0.9991,
        'TAU': 1,
        
        'STACK_FRAME': True,
        'STACK_SIZE': 4
    },
    
    'VizdoomCorridor-v0_Test' : {
        'ENV': 'VizdoomCorridor-v0',
        'RESOLUTION': [60, 80],
        'LAST_STEP': 2100,
        'WEIGHT_FILE': 'best_networks/VizdoomCorridor-v0_Best.pt',
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
        
        'LR': 1e-4,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 16,

        'UPDATE_METHOD': 'soft',
        'UPDATE_TARGET_EVERY': 80,
        'ALPHA': 0.01,

        'GAMMA': 0.99,
        
        'BUFFER_SIZE': 100000,

        'GREED_METHOD': 'epsilon',
        'EPSILON_START': 0.05,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY': 1.0, 
        'TAU': 1,
        
        'STACK_FRAME': True,
        'STACK_SIZE': 4
    },
    
    'VizdoomCorridor-v0_Kill' : {
        'ENV': 'VizdoomCorridor-v0',
        'RESOLUTION': [60, 80],
        'LAST_STEP': 2100,
        'WEIGHT_FILE': 'best_networks/VizdoomCorridor-v0_Kill.pt',
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
        
        'LR': 1e-4,
        'WEIGHT_DECAY': 1e-2,
        'BATCH_SIZE': 16,

        'UPDATE_METHOD': 'soft',
        'UPDATE_TARGET_EVERY': 80,
        'ALPHA': 0.01,

        'GAMMA': 0.99,
        
        'BUFFER_SIZE': 100000,

        'GREED_METHOD': 'epsilon',
        'EPSILON_START': 0.05,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY': 1.0, 
        'TAU': 1,
        
        'STACK_FRAME': True,
        'STACK_SIZE': 4
    },
    
    'VizdoomBasic-v0_Train' : {
        'ENV': 'VizdoomBasic-v0',
        'RESOLUTION': [120, 160],
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
        'WEIGHT_FILE': 'best_networks/VizdoomBasic-v0_Best.pt',
        'RESOLUTION': [120, 160],
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

        'GREED_METHOD': 'epsilon',
        'EPSILON_START': 0.05,
        'EPSILON_END': 0.05,
        'EPSILON_DECAY': 1.0, 
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
        return preprocess_vizdoom(state, getParam('RESOLUTION'))

    def _build_model(self, out_features):
        # Inspiré de : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    
        in_features = 1
        if getParam('STACK_FRAME'):
            in_features = getParam('STACK_SIZE')
    
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(getParam('RESOLUTION')[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(getParam('RESOLUTION')[1])))
        linear_input_size = convw * convh * 32
    
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
            torch.nn.Linear(linear_input_size, 256),
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
        
def conv2d_size_out(size, kernel_size = 5, stride = 2):
    return (size - (kernel_size - 1) - 1) // stride  + 1
        
        
        
        

# Boucle de train
def train(env, agent):
    total_steps = 0.
    episode_lengths = []
    episode_rewards = []

    # Itere les episodes
    reward, done = 0, False
    for ep in range(getParam('NB_TRAIN_EPISODE')):
        state = env.reset()
        
        if getParam('STACK_FRAME'):
            state = [state]*getParam('STACK_SIZE')

        sum_loss = 0.
        sum_step = 0.
        sum_reward = 0.
        
        # Itere les étapes
        for step in range(getParam('NB_MAX_TRAIN_STEP')):
            if getParam('RENDER_TRAIN'):
                env.render()
                
            action = agent.act(state, reward, done)
            next_state, reward, done, info = env.step(action)
            
            if getParam('STACK_FRAME'):
                next_state = state[1:] + [next_state]
            
            # Si on est sur la dernière étape possible
            # Pour pas avoir de 'done' à True avec une run perfect
            # On skip
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
        
    # Affichage du nuage de point final
    plt.scatter(list(range(getParam('NB_TRAIN_EPISODE'))), episode_rewards, label='Données')
    mean = sum(episode_rewards)/float(getParam('NB_TRAIN_EPISODE'))
    plt.plot(list(range(getParam('NB_TRAIN_EPISODE'))), [mean]*getParam('NB_TRAIN_EPISODE'), label='Moyenne (' + str(mean) + ')', linestyle='--')
    plt.legend(loc='lower right', framealpha=1).get_frame().set_edgecolor('black')
    plt.xlabel('Numéro épisode')
    plt.ylabel('Somme récompenses')
    plt.show()
        
        

# Boucle de test
def test(env, agent):
    total_steps = 0.
    episode_lengths = []
    episode_rewards = []

    # Itere les episodes
    reward, done = 0, False
    for ep in range(getParam('NB_TEST_EPISODE')):
        state = env.reset()

        if getParam('STACK_FRAME'):
            state = [state]*getParam('STACK_SIZE')
        
        sum_step = 0.
        sum_reward = 0.
        
        # Itere les étapes
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
        
    
    
    # Affichage du nuage de point final
    plt.scatter(list(range(getParam('NB_TEST_EPISODE'))), episode_rewards, label='Données')
    mean = sum(episode_rewards)/float(getParam('NB_TEST_EPISODE'))
    plt.plot(list(range(getParam('NB_TEST_EPISODE'))), [mean]*getParam('NB_TEST_EPISODE'), label='Moyenne (' + str(mean) + ')', linestyle='--')
    plt.legend(loc='lower right', framealpha=1).get_frame().set_edgecolor('black')
    plt.xlabel('Numéro épisode')
    plt.ylabel('Somme récompenses')
    plt.show()
        
        
        
        
        
if __name__ == '__main__':
    gym.logger.set_level(gym.logger.INFO)

    if IS_CART:
        # Creation de l'environnement
        env = gym.make(getParam('ENV'))
        if getParam('MONITOR'):
            env = gym.wrappers.Monitor(env, directory='./replay/cartpole', video_callable=lambda x: x%1 == 0, force=True)
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
            env = gym.wrappers.Monitor(env, directory='./replay/vizdoom', video_callable=lambda x: x%1 == 0, force=True)
        env.seed(0)
    
        # Creation de l'agent
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










