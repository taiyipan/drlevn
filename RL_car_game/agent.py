import torch
import random
import numpy as np
from collections import deque

from map_game import PlayerCar

from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 1_000_000
BATCH_SIZE = 10000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (should be smaller than 1, usually 0.8)
        self.memory = deque(maxlen=MAX_MEMORY) # when exceeding max memory --> popleft()
        self.model = Linear_QNet() # input, hidden, output
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        #TODO: model, trainer

    def preprocess_state(self, agent_vision):
        pass

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft is MAX_memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation 
        self.epsilon = 80 - self.n_games
        final_move = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 7)
            #final_move[move]
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            #final_move[move]

        return final_move[move]


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = PlayerCar(6,6)
    while True:
        # get old state
        state_old = game.get_agent_state()
        old = torch.from_numpy(state_old).unsqueeze(0)

        # get move
        final_move = agent.get_action((old))

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = torch.from_numpy(game.get_agent_state()).unsqueeze(0)
        # state_new = game.get_agent_state()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score  
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            

if __name__ == '__main__':
    #print(MAX_MEMORY)
    train()

##### ---- The Below code is for basic testing of game -----######
# run = True

# #images = [(GRASS,(0,0)), (TRACK,(0,0)), (FINISH, FINISH_POSITION), (TRACK_BORDER,(0,0))]
# player_car = PlayerCar(4, 4)

# while run:
     
#     player_car.play_step()
