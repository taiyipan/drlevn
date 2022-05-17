import pygame
import random
from game import SnakeGameAI

game = SnakeGameAI()

done = False
while True:
    final_move = [0,0,0]
    move = random.randint(0, 2)
    final_move[move] = 1
    reward, done, score = game.play_step(final_move)
    print(reward, done, score) 

    