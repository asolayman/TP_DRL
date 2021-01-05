import copy
from numpy.random import choice
from random import sample, randint, random
import torch


class DQNAgent():
    def __init__(self, out_features, lr=1e-3, batch_size=64, buffer_size=10000, update_method='soft', update_target_every=40, alpha=0.01,
                 gamma=0.9, greed_method='epsilon', epsilon_start=1.0, epsilon_end=1.0, epsilon_decay=0.995, tau=1, gpu=True, weight_file=None):
        
        # On set tout les attribus
        
        self.device = torch.device('cpu')
        if torch.cuda.is_available() and gpu:
            self.device = torch.device('cuda:0')
        
        print('Using :', self.device)
        
        self._build_model(out_features)
        if weight_file is not None:
            self.model.load_state_dict(torch.load(weight_file, map_location=lambda storage, loc: storage))
        self.model = self.model.to(self.device)
        self.target = copy.deepcopy(self.model)
            
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-1)
        self.loss_function = torch.nn.MSELoss()
            
        self.buffer_size = buffer_size
        self.buffer = []
        
        self.update_method = update_method
        self.update_target_every = update_target_every
        self.alpha = 0.01
        
        self.gamma = gamma
        self.greed_method = greed_method
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        
        
        
        
    def act(self, state, reward, done):
        # On recupere l'input qu'on unsqueeze pour simuler un batch
        input = torch.FloatTensor(self._preprocess_state(state)).unsqueeze(0).to(self.device)
        output = self.model(input)
        
        # Selon la méthod de greed on change ou pas l'action
        if self.greed_method is None:
            action = torch.argmax(output).item()
        elif self.greed_method == 'boltzmann':
            action = self._boltzmann_greed(output)
        elif self.greed_method == 'epsilon':
            action = self._epsilon_greed(output)
        else:
            action = torch.argmax(output).item()
        
        return action
        
       
       
    def train_step(self, total_steps):
        # On récupère le batch
        state_batch, action_batch, new_state_batch, reward_batch, dones_batch = self._get_batch(self.batch_size)
        
        # On calcul la prediction et la 'vraie' valeur
        q_pred = self.model(state_batch).gather(1, action_batch)
        q_truth = reward_batch + self.gamma*torch.amax(self.target(new_state_batch).detach(), 1).unsqueeze(1)*dones_batch
        
        # On calcul la loss + retro propagation
        loss = self.loss_function(q_pred, q_truth)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
    
        # Selon la méthode on update target ou pas
        if self.update_method == 'hard':
            if total_steps%self.update_target_every == 0:
                self._update_target()
        elif self.update_method == 'soft':
            self._soft_update_target()
        
        return loss.item()
    
    

    def add_interaction(self, state, action, next_state, reward, done):
        # On préprocess les états (utile avec doom)
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)
        
        # On ajoute au buffer, et si il faut on retire les vieilles interaction
        self.buffer.append((state, action, next_state, reward, done))
        while len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        
        
    def _update_target(self):
        self.target = copy.deepcopy(self.model)
        
        
        
    def _soft_update_target(self):
        # Update avec alpha, on itere les parameters et on applique la formule
        for (name, model_weight), (_, target_weight) in zip(self.model.named_parameters(), self.target.named_parameters()):
            if 'weight' in name or 'bias' in name:
                with torch.no_grad():
                    target_weight.data = (1-self.alpha)*model_weight.data + self.alpha*model_weight.data
        
        
    
    def _build_model(self, out_features):
        # Défini le réseau de neurone (utilisé par cartpole)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, out_features)
        )
        
        
        
    def _get_batch(self, size):
        sample = self._buffer_sample(size)
        
        # On passe sous pytorch + si il faut on change un peu le format du tensor
        state_batch = torch.FloatTensor([x[0] for x in sample]).to(self.device)
        action_batch = torch.LongTensor([x[1] for x in sample]).unsqueeze(1).to(self.device)
        new_state_batch = torch.FloatTensor([x[2] for x in sample]).to(self.device)
        reward_batch = torch.FloatTensor([x[3] for x in sample]).unsqueeze(1).to(self.device)
        dones_batch = 1.-torch.FloatTensor([x[4] for x in sample]).unsqueeze(1).to(self.device)

        return state_batch, action_batch, new_state_batch, reward_batch, dones_batch
        
        

    def _buffer_sample(self, size):
        # Si on tente de sample plus que la taille, on return ce qu'on a deja
        if size > len(self.buffer):
            return self.buffer
        return sample(self.buffer, size)
        
        
        
    def _preprocess_state(self, state):
        # Sert à être override pour doom
        return state

        
        
    def _epsilon_greed(self, output):
        best_action = torch.argmax(output).item()
    
        # Update epsilon
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end
        
        # Change ou pas l'action selon epsilon et le random
        if random() < self.epsilon:
            return randint(0, output.shape[1]-1)
        else:
            return best_action
    
    
    
    def _boltzmann_greed(self, output):
        # Calcul les proba
        prob = torch.exp(output/self.tau)
        prob = (prob/torch.sum(prob))[0].detach().cpu().numpy()
        
        # List des actions possibles
        list_action = list(range(output.shape[1]))
        
        # Choisi une action en fonction des probas
        action = choice(list_action, 1, p=prob)[0]
        
        return action








