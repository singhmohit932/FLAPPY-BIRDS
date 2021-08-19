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
    
    
    """
    Bird class representing the flappy bird
    """
    
    
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
        
        """
        make the bird jump
        :return: None
        """
        
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        
        """
        make the bird move
        :return: None
        """
        
        self.tick_count+=1
        
        # for downward acceleration
        displace_ment = self.velocity*self.tick_count + 1.5*self.tick_count**2
         
        # terminal velocity
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
        
        """
        draw the bird
        :param win: pygame window or surface
        :return: None
        """
        
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

        # so when bird is nose diving it isn't flapping
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2
         
        # tilt the bird
        rotated_image =  pygame.transform.rotate(self.img, self.tilt)
        new_rect =  rotated_image.get_react(center=self.img.get_react(top_left = (self.x,self.y)).center)
        win.blit(rotated_image, new_rect.top_left)

    
    def get_mask(self):
        """
        gets the mask for the current image of the bird
        :return: None
        """
        return pygame.mask.from_surface(self.img)



class Pipe:
    
    """
    represents a pipe object
    """
    
    GAP = 200
    VEL = 5


    def __init__(self,x):
        
        """
        initialize pipe object
        :param x: int
        :param y: int
        :return" None
        """
        
        
        
        self.x = x
        self.height = 0
         
        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0
        
        #storing the images of the pipe
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        # it is used for if the pipe has been passed . It is for the A.I. purposes and 
        # collision purpose
        self.passed = False
        
        # this for setting the height of the lower and upper pipe
        self.set_height()

    def set_height(self):
        
         """
        set the height of the pipe, from the top of the screen
        :return: None
        """
            
        self.height = random.randrange(50,450)
        
        #stores the top position of both the pipes
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
    

    def move(self):
        """
        move pipe based on vel
        :return: None
        """
        self.x -= self.VEL
    
    def draw(self, win):
        """
        draw both the top and bottom of the pipe
        :param win: pygame window/surface
        :return: None
        """
        # draw top
        win.blit(self.PIPE_TOP,(self.x,self.top))
        # draw bottom
        win.blit(self.PIPE_BOTTOM, (self.x,self.bottom))

    def collide(self, bird):
        """
        returns if a point is colliding with the pipe
        :param bird: Bird object
        :return: Bool
        """
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
    """
    Represnts the moving floor of the game
    """
    VEL = 5
    IMG = BASE_IMG
    WIDTH = BASE_IMG.get_width()


    
    def __init__(self, y):
        """
        Initialize the object
        :param y: int
        :return: None
        """
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH


    def draw(self,win):
        """
        Draw the floor. This is two images that move together.
        :param win: the pygame surface/window
        :return: None
        """
        win.blit(self.IMG,(self.x1,self.y))
        win.blit(self.IMG,(self.x2,self.y))




def  draw_window(win, birds, pipes, base, score):
    """
    draws the windows for the main game loop
    :param win: pygame window surface
    :param bird: a Bird object
    :param pipes: List of pipes
    :param score: score of the game (int)
    :param gen: current generation
    :param pipe_ind: index of closest pipe
    :return: None
    """
    
    win.blit(BG_IMG, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    
    text = STAT_FONT.render("Score: " +  str(score), 1,(255,255,255))
    win.blit(text,(WIN_WIDTH - 10 - text.get_width(),10))
    for bird in birds:
       bird.draw(win)
    pygame.display.update()

def eval_genomes(genomes, config):
    
    """
    runs the simulation of the current population of
    birds and sets their fitness based on the distance they
    reach in the game.
    """
    
    global WIN 
    nets = []
    ge = []
    birds =  []
    

    for g in genomes :
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness = 0 
        ge.append(g)

      
    base = BASE(730)
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
        draw_window(win,birds,pipes,base,score)   



def run(config_path):
    
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """

     config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
     
     # Create the population, which is the top-level object for a NEAT run.
     p = neat.Population(config)
       
     # Add a stdout reporter to show progress in the terminal.
     p.add_reporter(neat.StdOutReporter(True))
     stats = neat.statisticsReporter()
     p.add_reporter(stats)
     #p.add_reporter(neat.Checkpointer(5))
     
     # Run for up to 50 generations.
     winner = p.run(eval_genomes, 50)

     # show final stats
     print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "main":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config_feedforward.txt")
    run(config_path)




