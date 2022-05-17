import pygame
import math
import random
import cv2
import numpy as np
from rotated_rect_crop import crop_rotated_rectangle

def scale_image(img, factor):
    size = round(img.get_width()* factor), round(img.get_height()* factor)
    return pygame.transform.scale(img, size)

def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    rows, cols = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    out = cv2.getRectSubPix(img_rot, size, center)

    return out, img_rot

class PlayerCar:
    START_POS = (500,500) 
    
    def __init__(self, max_vel, rotation_vel):
        pygame.display.set_caption("RACING GAME!")
        self.clock = pygame.time.Clock()

        
        self.TRACK = scale_image(pygame.image.load("imgs/indoor_maps/map1.png"),1.0)
        self.TRACK_BORDER = scale_image(pygame.image.load("imgs/indoor_maps/mask1.png"),2.0)
        self.TRACK_BORDER_MASK = pygame.mask.from_surface(self.TRACK_BORDER)
        self.WIDTH, self.HEIGHT = self.TRACK.get_width(), self.TRACK.get_height()
        self.GIFT = pygame.image.load("imgs/gift.png")
        self.GIFT_MASK = pygame.mask.from_surface(self.GIFT)
        self.gift_pos_x, self.gift_pos_y = random.randint(0,self.WIDTH), random.randint(0,self.HEIGHT)
        self.GIFT_POSITION = (self.gift_pos_x, self.gift_pos_y)
        self.PURPLE_CAR = scale_image(pygame.image.load("imgs/purple-car.png"), 0.5)

        self.agent_vision_width, self.agent_vision_height = 256,256

        self.IMAGES = [(self.TRACK,(0,0)), (self.TRACK_BORDER,(0,0))]

        self.win = pygame.display.set_mode((self.WIDTH,self.HEIGHT))
        color = (255, 255, 255) # Initialing RGB Color 
        self.win.fill(color) # Changing surface color

        self.IMG = self.PURPLE_CAR

        self.images = self.IMAGES
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
        self.score = 0


    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def place_gift(self):
        #self.gift_pos_x, self.gift_pos_y = random.randint(0,self.WIDTH), random.randint(0,self.HEIGHT)
        self.win.blit(self.GIFT, self.GIFT_POSITION)
        self.draw()
        pygame.display.update()

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
        # self.x, self.y = self.START_POS
        # self.angle = 0
        # self.vel = 0 
        self.gift_pos_x, self.gift_pos_y = random.randint(0,self.WIDTH), random.randint(0,self.HEIGHT)
        self.GIFT_POSITION = (self.gift_pos_x, self.gift_pos_y)
        self.place_gift()

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration/2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel
        self.move()

    def move_player(self, action):
        keys = pygame.key.get_pressed()
        moved = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break           

        # actions = forward    
        # actions = [1,0,0,0], [0,1,0,0], [0,0,1,0], 
        if action[0] == 1:
            self.move_forward()
        if action[1] == 1:
            self.move_backward()
        if action[2] == 1:
            self.rotate(left=True)
        if action[3] == 1:
            self.rotate(right=True)

        # if keys[pygame.K_a]:
        #     self.rotate(left=True)
        # if keys[pygame.K_d]:
        #     self.rotate(right=True)
        # if keys[pygame.K_w]:
        #     moved = True
        #     self.move_forward()
        # if keys[pygame.K_s]:
        #     moved = True
        #     self.move_backward()

        if not moved:
            self.reduce_speed()

    def handle_collision(self):
        reward = 0
        don = False
        if self.collide(self.TRACK_BORDER_MASK) != None:
            self.bounce() # This is collision with borders and track
            reward = -20
            don = True

        player_GIFT_poi_collide = self.collide(self.GIFT_MASK, *self.GIFT_POSITION)
        if player_GIFT_poi_collide != None:
            self.reset() # This collision is for collision with GIFT
            reward = 50
            don = False
        return reward, don

    def play_step(self,action):
        self.move_player(action)
        rew, done = self.handle_collision()
        self.score += rew
        for im, pos in self.images:
            self.win.blit(im, pos)
            self.draw()
        self.place_gift()
        self.clock.tick(60)
        #print(round(self.x,0), round(self.y,0))
        return rew, done, self.score

    def get_agent_state(self):
        x0 = int(self.x)
        y0 = int(self.y)
        theta = int(self.angle)
        # print(theta)
        s, c = np.cos(theta), np.sin(theta)

        R = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        p1 = np.matmul(R, np.array([[x0 - 128], [y0], [0]]))
        x1, y1 = p1[0][0], p1[1][0] 

        p2 = np.matmul(R, np.array([[x0 - 128], [y0 + 256], [0]]))
        x2, y2 = p2[0][0], p2[1][0] 

        p3 = np.matmul(R, np.array([[x0 + 128], [y0 + 256], [0]]))
        x3, y3 = p3[0][0], p3[1][0] 

        p4 = np.matmul(R, np.array([[x0 + 128], [y0], [0]]))
        x4, y4 = p4[0][0], p4[1][0] 

        center = (x0, y0)
        width, height = self.agent_vision_width, self.agent_vision_height
        rect = (center,(height, width), theta) 

        pygame.display.update()
        hwc_game_state = pygame.surfarray.array3d(pygame.display.get_surface())
        hwc_game_state = cv2.rotate(hwc_game_state,cv2.cv2.ROTATE_90_CLOCKWISE)
        hwc_game_state = cv2.flip(hwc_game_state,1)
        
        cropped_img = cv2.resize(hwc_game_state,(256,256))
        agent_perception = crop_rotated_rectangle(image = hwc_game_state, rect = rect) #256,256,3
        #agentperception = cv2.rotate(agent_perception,90)
        #chw_game_state = hwc_game_state.transpose((2,0,1))


        return agent_perception.reshape(3,256,256)

if __name__=="__main__":
    player_car = PlayerCar(6, 6)
    run = True
    while run:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            
        reward, done, score = player_car.play_step()
        #print(reward, done, score)
        agent_view = player_car.get_agent_state()
        cv2.imshow('agent display',agent_view)

    pygame.quit()
