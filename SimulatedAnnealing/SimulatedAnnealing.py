import numpy as np
import math
import random

import pygame
from environment import VectorizedEnv
from ia import save_ai,load_ai

class SimulatedAnnealing:
    def __init__(self,EnvClass,time_max = 2000 ,  alpha : float = 0.9999, iterations : int = 100,t0=1000) -> None:
        
        self.iteration = 0
        self.bestScore = float('-inf')
        self.EnvClass = EnvClass
        # si on montre l'environnement
        # pygame.init()
        # self.font = pygame.font.SysFont(None, 30)
        # self.screen = pygame.display.set_mode(EnvClass.display_size)
        # self.clock = pygame.time.Clock()
        # self.FPS = EnvClass.render_fps
        self.time_max = time_max
        self.envs = EnvClass(screen=None, time_max=self.time_max)
        self.brains = np.random.uniform(-1,1, size=(time_max, self.envs.action_size)) 
        self.alpha = alpha
        self.iterations = iterations
        self.t0 = t0
        self.t = self.t0
        self.bestBrains = self.brains

        self.scoresListe = []
        self.iterationsListe= []

    def temperature(self,t):
        return self.t0/math.log(1 + t + 0.0000001)
    
    def cooling(self,t):
        return self.alpha * t

    def objective_function(self,brains):
        obs, _ = self.envs.reset()
        done = False
        total_scores = 0

        while not done:
            actions = brains[int(obs[-1]%len(brains))] 

            obs, rewards, done, _ = self.envs.step(actions)
            total_scores += rewards

        return total_scores
    
    def train(self,show = False):
        currentScore = self.objective_function(self.brains)
        self.bestScore = currentScore
        for iteration in range(self.iterations):
            self.iteration = iteration
            newBrains = np.clip(self.brains + np.random.normal(0,1,size=self.brains.shape),a_min=-1, a_max=1)  
            newScore = self.objective_function(newBrains)
            delta = currentScore - newScore
            
            acceptance_prob = np.clip(math.exp((currentScore-newScore)/self.t),0,1)
            
            if newScore < currentScore or acceptance_prob > random.uniform(0, 1):
                self.brains = newBrains
                currentScore = newScore
                self.scoresListe.append(currentScore)
                self.iterationsListe.append(iteration)


            # Mettre à jour la meilleure solution trouvée
            if currentScore > self.bestScore:
                self.bestBrains = self.brains
                self.bestScore = currentScore
                print("Iterations, Best Score, Temperatures : ",iteration, np.amax(self.bestScore), self.t)

                temp = self.envs
                self.envs = None
                save_ai(self, f"./09999v2/iter{iteration}-{self.bestScore}")
                self.envs = temp
             # Mettre à jour la température
            self.t = self.cooling(self.t)



    def test(self, show=False):
        pygame.init()
        font = pygame.font.SysFont(None, 30)
        screen = pygame.display.set_mode(self.EnvClass.display_size)
        clock = pygame.time.Clock()
        FPS = self.EnvClass.render_fps

        env = self.EnvClass(screen = screen,  time_max=self.time_max)
        obs, _ = env.reset()
        done = False
        total_scores = 0

        while not done:

            actions = self.bestBrains[int(obs[-1]%len(self.bestBrains))] 
            obs, rewards, done, _ = env.step(actions)
            total_scores -= rewards


            for e in pygame.event.get():
                if e.type is pygame.QUIT:
                    exit(0)

            env.render()
          
            iter_text = font.render(f"Itération actuelle: {self.iteration} ", True, (0,0,0))
            score_text = font.render(f"Score : {-total_scores}", True, (0,0,0))
            bestScoreGlob_text = font.render(f"Meilleur score: {-self.bestScore}", True, (0,0,0))
            
            screen.blit(iter_text, (10,10))
            screen.blit(score_text, (10, 30))
            screen.blit(bestScoreGlob_text, (10, 50))

            pygame.display.flip()
            clock.tick(FPS)
            pygame.display.set_caption(f"Exemple Vectorized {self.EnvClass.__name__} (fps: {clock.get_fps():.0f})")
