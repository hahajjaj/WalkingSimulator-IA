
import copy
import os
from typing import Type
import numpy as np
import pygame

from environment import IEnvironment, VectorizedEnv, WalkingEnv
from concurrent import futures

class HillClimbing:


    def __init__(
            self, 
            EnvClass: Type[IEnvironment] = WalkingEnv,
            epsilon: float = 0.000001,
            acceleration: float = 1.2,
            time_max: int = 1000,
            step_size = 1.0, 
            max_iter = 100,
            num_matrix: int = 1000,
            envKwargs: dict = {}, # Arguments concernant l'environnement
        ):

        self.EnvClass = EnvClass
        self.env = self.EnvClass(**envKwargs, time_max=time_max)

        self.step_size = step_size
        self.max_iter = max_iter
        self.num_matrix = num_matrix

        self.time_max = time_max

        self.initialPoint = np.zeros((self.time_max, self.env.action_size))
        self.initialStepSizes = np.ones((self.time_max, self.env.action_size)) # a verifier

        self.acceleration = acceleration

        self.epsilon = epsilon

        self.bestSolution = None

    # def create_point(self) -> np.ndarray:
    #     ...

    def _eval(self, solutions: np.ndarray, show: bool = False) -> float:
        states, _ = self.envs.reset()

        scores = np.zeros(4)
        dones = [False for _ in range(4)]

        while not all(dones): # Simulation de chaque solutions

            actions = [
                solution[int(states[i][-1]) % self.time_max] for i, solution in enumerate(solutions)]
            states, rewards, dones, _ = self.envs.step(actions)
            scores += rewards

            if not show: continue

            for e in pygame.event.get():
                if e.type is pygame.QUIT:
                    exit(0)

            self.envs.render()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        return scores

    def train(self, show: bool = False):
        
        if show:
            pygame.init()
            self.screen = pygame.display.set_mode(self.EnvClass.display_size)
            self.clock = pygame.time.Clock()
            self.FPS = self.EnvClass.render_fps
        else:
            self.screen = None

        self.envs = VectorizedEnv(self.EnvClass, 4, screen=self.screen, time_max=self.time_max)

        stepSize = self.initialStepSizes
        candidate = [
            -self.acceleration,
            -1 / self.acceleration,
            1 / self.acceleration,
            self.acceleration
        ]
        bestScore= 0.0
        iteration = 0
        overalBestScore = 0.0
        for l in range(100):

            currPoint = np.random.rand(self.time_max, self.env.action_size)
            bestScore = self._eval([currPoint,], show)[0]

            print("Test: ", l, "Overall best score: ", overalBestScore)
            while True:
                iteration += 1
                print("Iteration: ", iteration)
                beforeScore = bestScore
                for k in range(len(currPoint)):
                    for i in range(len(currPoint[k])):
                        beforePoint = currPoint[k][i]
                        bestStep = 0.0
                        
                        steps = [stepSize[k][i] * c for c in candidate]
                        points = [np.array(currPoint, copy=True) for _ in range(4)]

                        for j, step in enumerate(steps):
                            points[j][k][i] = beforePoint + step

                        scores = self._eval(points, show)
                        for j, score in enumerate(scores):
                            if score > bestScore:
                                bestScore = score
                                bestStep = steps[j]
                                print("New best score: ", bestScore)

                        # if score > bestScore: # Si on a trouvé une meilleure solution, on la met à jour 
                        #     bestScore = score
                        #     bestStep = step
                        #     print("New best score: ", bestScore)

                        if bestStep == 0: # Si aucune solution a été trouvée 
                            currPoint[k][i] = beforePoint
                            stepSize[k][i] = stepSize[k][i] / self.acceleration
                        else:
                            currPoint[k][i] = beforePoint + bestStep
                            stepSize[k][i] = bestStep // self.acceleration
                            
                if (bestScore - beforeScore) < self.epsilon:
                    if bestScore > overalBestScore:
                        overalBestScore = bestScore
                        self.bestSolution = currPoint
                    print("Best solution: ", self.bestSolution)
                    # return
                    break

    
    # def train2(self, show: bool = False):
    #     self.env.setVisualization(show)


    #     currPoint = self.initialPoint
    #     stepSize = self.initialStepSizes
    #     candidate = [
    #         -self.acceleration,
    #         -1 / self.acceleration,
    #         1 / self.acceleration,
    #         self.acceleration
    #     ]
    #     bestScore = self._eval([currPoint])[0]

    #     iteration = 0

    #     while True:
    #         iteration += 1
    #         print("Iteration: ", iteration)
    #         beforeScore = bestScore
    #         for k in range(len(currPoint)):

    def objective_function(self, solution):

        env = self.EnvClass(screen=None)

        done = False
        obs, _ = env.reset()

        score = 0.0
        while not done:
            action = solution[int(obs[-1] % self.time_max)]
            obs, reward, done, _ = env.step(action)
            score += reward
        
        return score

    def hill_climbing(self, start_matrix, step_size, epsilon):
        best_matrix = start_matrix.copy()
        best_score = self.objective_function(start_matrix)
        while True:
            before_score = best_score
            # Generate a new candidate solution by randomly perturbing the current solution
            matrix = start_matrix + np.random.normal(loc=0.0, scale=step_size, size=start_matrix.shape)
            matrix = np.clip(matrix, -1, 1)
            score = self.objective_function(matrix)
            # Update the current solution if the new candidate is better
            if score > best_score:
                best_matrix = matrix.copy()
                best_score = score

            if (best_score - before_score) < epsilon:
                break
        return best_matrix, best_score

    @staticmethod
    def hill_climbing_multi(start_matrix, step_size, epsilon, objective_function, worker_id, EnvClass, time_max):
        best_matrix = start_matrix.copy()
        best_score = objective_function(start_matrix, EnvClass, time_max)
        while True:
            before_score = best_score
            # Generate a new candidate solution by randomly perturbing the current solution
            matrix = start_matrix + np.random.normal(loc=0.0, scale=step_size, size=start_matrix.shape)
            matrix = np.clip(matrix, -1, 1)
            score = objective_function(matrix, EnvClass, time_max)
            # Update the current solution if the new candidate is better
            if score > best_score:
                best_matrix = matrix.copy()
                best_score = score
                # print("Worker:", worker_id, "New best score:", best_score)

            if (best_score - before_score) < epsilon:
                break
        return best_matrix, best_score

    @staticmethod
    def eval(solution, EnvClass, time_max):
        env = EnvClass(screen=None)

        done = False
        obs, _ = env.reset()

        score = 0.0
        while not done:
            action = solution[int(obs[-1] % time_max)]
            obs, reward, done, _ = env.step(action)
            score += reward
        return score
    

    def train(self, show: bool = False):
        # Choose multiple starting matrices
        env = self.EnvClass(screen=None, time_max=self.time_max)
        start_matrices = [np.random.uniform(low=-1, high=1, size=(self.time_max, env.action_size)) for _ in range(self.num_matrix)]

        # Run the hill climbing algorithm from each starting matrix

        with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(self.hill_climbing_multi, *zip(*[
                (start_matrix, self.step_size, self.epsilon, self.eval, worker_id, self.EnvClass, self.time_max)
                for worker_id, start_matrix in enumerate(start_matrices)
            ]))
            # results = executor.map(self.t, range(10), range(10))
        best_solution, best_score = None, float('-inf')
        for worker_id, result in enumerate(results):
            solution, score = result
            print(f"Worker {worker_id} done score {score}")
            if score > best_score:
                best_solution, best_score = solution, score
                print("New best score:", best_score)
        # best_solution, best_score = max(results, key=lambda x: x[1])

        # for start_matrix in start_matrices:
        #     solution, score = self.hill_climbing(start_matrix, step_size=self.step_size, max_iter=self.max_iter)
        #     if score > best_score:
        #         best_solution, best_score = solution, score
        #         print("New best score:", best_score)

        # Print the best solution found
        print("Best solution:")
        print(best_solution)
        print("Score:", best_score)

                    
    def test(self, show: bool = False):
        self.envs.setVisualization(show)

        states, _, dones, _ = self.envs.reset()

        while not all(dones): # Simulation de chaque solutions 
            actions = [self.bestSolution[int(states[0][-1])]]
            states, scores, dones, _ = self.envs.step(actions)

    def save(self, path: str):
        np.save(path, self.bestSolution)

    def load(self, path: str):
        self.bestSolution = np.load(path)
