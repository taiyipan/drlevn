import pygame
import time
import math

def scale_image(img, factor):
    size = round(img.get_width()* factor), round(img.get_height()* factor)
    return pygame.transform.scale(img, size)


GRASS = scale_image(pygame.image.load("imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("imgs/track.png"),0.9)

TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"),0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"),0.5)

WIDTH,HEIGHT =TRACK.get_width(), TRACK.get_height()

class PlayerCar:
    IMG = RED_CAR
    START_POS = (180,200)
    IMAGES = [(GRASS,(0,0)), (TRACK,(0,0)), (FINISH, FINISH_POSITION), (TRACK_BORDER,(0,0))]

    def __init__(self, max_vel, rotation_vel):
        pygame.display.set_caption("RACING GAME!")
        self.win = pygame.display.set_mode((WIDTH,HEIGHT))
        self.images = self.IMAGES
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self):
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        self.win.blit(rotated_image, new_rect.topleft)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        self.move()
    
    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal
    
    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0 

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration/2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel
        self.move()

    def move_player(self):
        keys = pygame.key.get_pressed()
        moved = False

        if keys[pygame.K_a]:
            self.rotate(left=True)
        if keys[pygame.K_d]:
            self.rotate(right=True)
        if keys[pygame.K_w]:
            moved = True
            self.move_forward()
        if keys[pygame.K_s]:
            moved = True
            self.move_backward()

        if not moved:
            self.reduce_speed()

    def handle_collision(self):
        if self.collide(TRACK_BORDER_MASK) != None:
            self.bounce()

        player_finish_poi_collide = self.collide(FINISH_MASK, *FINISH_POSITION)
        if player_finish_poi_collide != None:
            if player_finish_poi_collide[1] == 0:
                self.bounce()
            else:
                self.reset()
                print("finish")

        
    def play_step(self):
        self.move_player()
        self.handle_collision()

        for im, pos in self.images:
            self.win.blit(im, pos)
            self.draw()

        pygame.display.update()
            

run = True

#images = [(GRASS,(0,0)), (TRACK,(0,0)), (FINISH, FINISH_POSITION), (TRACK_BORDER,(0,0))]
player_car = PlayerCar(4, 4)

while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break
        
    player_car.play_step()


pygame.quit()
