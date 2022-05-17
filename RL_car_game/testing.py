import pygame
import random
from map_game import PlayerCar

game = PlayerCar(6,6)

done = False
while True:
    final_move = [0,0,0,0]
    move = random.randint(0, 3)
    final_move[move] = 1
    reward, done, score = game.play_step(final_move)
    print(reward, done, score) 

    