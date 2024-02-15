from functools import partial
import math
import os
import pickle
import time
from typing import Optional, Type
# import cv2
import numpy as np
from neat import nn, population, StatisticsReporter, StdOutReporter
import neat
# import visualize
import pygame
from consts import NEAT_TRAIN_CONFIG_FORMAT
from environment import IEnvironment, VectorizedEnv, VectorizedWalkingEnv
from ia import IAI
from screen import Window





class NEAT:
    

    def __init__(self, EnvClass: Type[IEnvironment], gen_max = 100 , pop_size = 30, selection_rate=0.3, conn_rate=0.5, node_rate=0.2, **env_kwargs) -> None:
        self.pop_size = pop_size
        self.nb_gen = gen_max
        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        self.selection_rate = selection_rate
        self.conn_rate = conn_rate
        self.node_rate = node_rate

        self.best_genome = None
        self.best_score = float('-inf')
        self.gen_counter = 0

        self.config = None

        self.stats = StatisticsReporter()
        self.elapse_time_reporter = ElapseTimeReporter()


    @staticmethod
    def load_config(pop_size, observation_size, action_size, conn_rate, node_rate, selection_rate):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'walking_neat_config')
        config_str = NEAT_TRAIN_CONFIG_FORMAT.format(pop_size=pop_size, num_inputs=observation_size, num_outputs=action_size, conn_rate=conn_rate, node_rate=node_rate, selection_rate=selection_rate)
        with open(config_path, 'w') as f:
            f.write(config_str)

        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

    def train(self, show = False, check_point: Optional[str] = None) -> None:
        
        # if show: # setup rendering if asked
        #     window = Window(self.EnvClass.display_size, self.EnvClass.render_fps, f"NEAT train on {self.EnvClass.__name__} (fps: %fps%)")
        #     screen = window.screen
        # else:
        #     screen = None

        envs = VectorizedWalkingEnv(self.EnvClass, self.pop_size, None, **self.env_kwargs)

        self.config = self.load_config(self.pop_size, envs.observation_size, envs.action_size, self.conn_rate, self.node_rate, self.selection_rate)
        self.pop = population.Population(self.config)
        # self.gen_counter = 0

        def fitness_function(genomes, neat_config):
            N_GENOME = len(genomes)
            
            # self.gen_counter += 1
            # print(f"\n***** Generation {self.gen_counter} *****")
            # start_time = time.time()

            envs = VectorizedWalkingEnv(self.EnvClass, N_GENOME, None, **self.env_kwargs)

            nets = [nn.FeedForwardNetwork.create(genome, neat_config) for _, genome in genomes]

            obss, _  = envs.reset()
            dones = [False for _ in range(N_GENOME)]
            scores = np.zeros(N_GENOME)

            # runs the simulation
            while not all(dones):
                actions = [list(map(math.tanh, nn.activate(obs))) for nn, obs in zip(nets, obss)]

                obss, rewards, dones, _ = envs.step(actions)
                scores += rewards

                # if not show: continue

                # best_genome_index = np.argmax(scores)

                # # envs.render([best_genome_index])
                # envs.render(np.argsort(scores)[::-1])
                # window.add_text(f"Meilleur score: {self.best_score:.0f}", (10, 10))
                # window.add_text(f"Score: {scores[best_genome_index]:.0f}", (10, 35))
                # window.render()

            # updates genomes's score
            for fitness, (_, genome) in zip(scores, genomes):
                genome.fitness = fitness

            # best_genome_index = np.argmax(scores)
            # best_score_of_gen = scores[best_genome_index]
            
            # # keeps tracks of best genome over the generations
            # if best_score_of_gen > self.best_score:
            #     # print("!!! New best score !!!")
            #     self.best_score = best_score_of_gen
            #     self.best_genome = genomes[best_genome_index][1]

                # if asked create a check point file
                # if check_point is not None:
                #     self.save(check_point + f"_{self.gen_counter}-{self.best_score:.0f}")
            
            # print(f"Best score : {best_score_of_gen:.1f}")
            # print(f"Time: {time.time() - start_time:.1f} s")
        if getattr(self, "elapsed_time_reporter", None) is not None:
            self.pop.add_reporter(self.elapse_time_reporter)
        self.pop.add_reporter(self.stats)
        self.pop.add_reporter(StdOutReporter(True))
        
        # starts the neat algorithm
        print("Starting NEAT algorithm with parameters:")
        print(f"Population size: {self.pop_size}")
        print(f"Number of generations: {self.nb_gen}")
        self.best_genome = self.pop.run(fitness_function, n=self.nb_gen)

        # visualize.plot_stats(self.stats, view=True)
        # visualize.plot_species(self.stats, view=True)
        # visualize.draw_net(self.config, self.best_genome, view=True)

    
    def test(self, show: bool = False, video: str = None):
        if self.best_genome is None:
            print("No best genome found. Try training one first or loading one from a save")
            return
        
        if show:
            window = Window(self.EnvClass.display_size, self.EnvClass.render_fps, f"NEAT test on {self.EnvClass.__name__} (fps: %fps%)")
            screen = window.screen
        else:
            screen = None

        # if video:
        #     out = cv2.VideoWriter(video+".mp4",cv2.VideoWriter_fourcc(*'mp4v'),self.EnvClass.render_fps, self.EnvClass.display_size)

        env = self.EnvClass(screen=screen, **self.env_kwargs)

        net = nn.FeedForwardNetwork.create(self.best_genome, self.config)

        obs, _ = env.reset()
        done = False
        score = 0.0

        while not done:
            action = list(map(math.tanh, net.activate(obs)))
            obs, reward, done, _ = env.step(action)
            score += reward

            if not show: continue

            env.render()
            window.add_text(f"Score: {score:.0f}")
            window.render()

            # if video:
            #     frame = pygame.surfarray.array3d(screen)

            #     frame = frame.transpose([1, 0, 2])

            #     img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #     out.write(img_bgr)

        # if video:
        #     cv2.destroyAllWindows()
        #     out.release()

        

    def save(self, file_path: str):
        print(f"Saving {self.__class__.__name__} to '{file_path}'...", end='')
        with open(file_path, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done")

    @classmethod
    def load(cls, file_path: str) -> 'NEAT':
        print(f"Loading {cls.__name__} from '{file_path}'...", end='')
            
        with open(file_path, 'rb') as file:
            ai = pickle.load(file)

        print('Done')

        return ai


class ElapseTimeReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.times = []

    def start_generation(self, generation):
        self.start_time = time.time()

    def post_evaluate(self, config, population, species, best_genome):
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)
        self.start_time = time.time()