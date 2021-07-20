import pygame
import random
import os
import time
import neat
import visualize
import pickle
from setuptools.command.setopt import config_file
from PIL import Image
pygame.font.init()



WIN_WIDTH  = 600
WIN_HEIGHT = 800

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")


PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
BG_IMG = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

# # BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))),
# #              pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),
# #              pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]
# # BASE_IMG = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))]
# # BG_IMG = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bg.png")))]
# # PIPE_IMG = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))]

STAT_FONT = pygame.font.SysFont("comicsans", 50)

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5


    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count+=1

        displace_ment = self.velocity*self.tick_count + 1.5*self.tick_count**2

        if displace_ment>=16:
            displace_ment=16

        if displace_ment < 0:
            displace_ment-=2
        
        self.y = self.y + displace_ment

        if displace_ment < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                   self.tilt -= self.ROT_VEL


    def draw(self, win):
        self.img_count += 1


        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0 


        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        rotated_image =  pygame.transform.rotate(self.img, self.tilt)
        new_rect =  rotated_image.get_react(center=self.img.get_react(top_left = (self.x,self.y)).center)
        win.blit(rotated_image, new_rect.top_left)


    def get_mask(self):
        return pygame.mask.from_surface(self.img)



class Pipe:
    GAP = 200
    VEL = 5


    def __init__(self,x):
        self.x = x
        self.height = 0
       
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG


        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
    

    def move(self):
        self.x -= self.velocity
    
    def draw(self, win):
        win.blit(self.PIPE_TOP,(self.x,self.top))
        win.blit(self.PIPE_BOTTOM, (self.x,self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)


        top_offset = (self.x - bird.x,self.top - round(bird.y))
        bottom_offset = (self.x - bird.x,self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)
        

        if t_point or b_point:
            return True

        return False


class BASE:
    VEL = 5
    IMG = BASE_IMG
    WIDTH = BASE_IMG.get_width()



    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.velocity
        self.x2 -= self.velocity

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH


    def draw(self,win):
        win.blit(self.IMG,(self.x1,self.y))
        win.blit(self.IMG,(self.x2,self.y))




def  draw_window(win, birds, pipes, base, score):
    win.blit(BG_IMG, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    
    text = STAT_FONT.render("Score: " +  str(score), 1,(255,255,255))
    win.blit(text,(WIN_WIDTH - 10 - text.get_width(),10))
    for bird in birds:
       bird.draw(win)
    pygame.display.update()

def main(genomes, config):
    global WIN 
    nets = []
    ge = []
    birds =  []
    

    for _,g in genomes :
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness = 0 
        ge.append(g)

      
    base = base(730)
    win = WIN
    clock = pygame.time.clock()
    pipes = [Pipe(600)]
    
    score = 0
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

            # bird.move()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind=1
        else:
            run = False
            break


        
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1


            output = nets[x].activate((bird.y, abs(bird.y- pipes[pipe_ind].height), abs(bird.y- pipes[pipe_ind].bottom)))
              
            if output[0] > 0.5:
                bird.jump()






        add_pipe = False
        rem = []
        for pipe in pipes:
            for x,bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness-=1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)




                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True


            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()
           
        if add_pipe:
            score+=1
            for g in ge:
                g.fitness+=5

            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)
            
        for x,bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
               birds.pop(x)
               nets.pop(x)
               ge.pop(x)


        base.move()
        draw_window(win,birds,pipes,base)   



def run(config_path):

     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

     p = neat.Population(config)

     p.add_reporter(neat.StdOutReporter(True))
     stats = neat.statisticsReporter()
     p.add_reporter(stats)

     winner = p.run(main,50)


if __name__ == "main":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config_feedforward.txt")
    run(config_path)




