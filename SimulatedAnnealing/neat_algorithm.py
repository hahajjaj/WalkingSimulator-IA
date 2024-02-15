import math
import os
import numpy as np
from neat import nn, population
import neat
import pygame
from environment import VectorizedEnv

class NEAT:
    def __init__(self, EnvClass, nb_gen = 10 , pop_size = 50) -> None:
        self.pop_size = 2
        self.nb_gen = 5
        self.env = EnvClass
        self.gen_counter = 0
       

    def evaluate(self, genomes, neat_config) -> np.ndarray:
        #pygame.init()
        #screen = pygame.display.set_mode(self.env.display_size)
        #clock = pygame.time.Clock()
        #FPS = self.env.render_fps
        self.gen_counter += 1 
        self.envs = VectorizedEnv(self.env, len(genomes), None)
        nets = []  
        for genome_id, genome in genomes:
            net = nn.FeedForwardNetwork.create(genome, neat_config)   
            nets.append(net)


        obss, _ = self.envs.reset()
        dones = [False for _ in range(len(genomes))]
        total_rewards = np.zeros(len(genomes))

        while not all(dones):
            actions = [list(map(math.tanh, nn.activate(obs))) for nn, obs in zip(nets, obss)]
            
            obss, rewards, dones, _ = self.envs.step(actions)
            total_rewards += rewards

            #for e in pygame.event.get():
             #   if e.type is pygame.QUIT:
            #        exit(0)
                
        
            #self.envs.render([0])
            #pygame.display.flip()
            #clock.tick(FPS)
            #pygame.display.set_caption(f"fps: {clock.get_fps():.0f}")
        
        for reward, (genome_id, genome) in zip(total_rewards, genomes):
            genome.fitness = reward

        
        print("Génération numéro : ", self.gen_counter)
        print("Best score : ", max(total_rewards))

    def train(self, show = False) -> None:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'walking_neat_config')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        pop = population.Population(config)
        pop.run(self.evaluate, n=self.nb_gen)



