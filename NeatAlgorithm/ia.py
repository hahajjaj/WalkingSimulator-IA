import pickle
import random
import time
from typing import Protocol, Type
import statistics as stats

import pygame
from environment import IEnvironment, VectorizedEnv, WalkingEnv

import numpy as np

def loadingBar(done: int, total: int, length: int = 20):
    return f"[{'=' * (done * length // total)}{' ' * (length - done * length // total)}] {done}/{total}" + ('\n' if done == total else '\r')

class IAI(Protocol):

    def __init__(self, EnvClass: Type[IEnvironment], **kwargs):
        ...

    def train(self, show: bool = False):
        ...

    def test(self, show: bool = False):
        ...

    def save(self, saveFile: str):
        ...

    @classmethod
    def load(cls, saveFile: str):
        ...

def save_ai(ai: IAI, file_path: str):
    print(f"Saving {ai.__class__.__name__} to '{file_path}'...", end='')

    with open(file_path, 'wb') as file:
        pickle.dump(ai, file, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done")

def load_ai(file_path: str) -> IAI:
    print(f"Loading AI from '{file_path}'...", end='')
        
    with open(file_path, 'rb') as file:
        ai = pickle.load(file)

    print('Done')

    return ai

class GeneticAlgo:
    def __init__(
            self,
            EnvClass: Type[IEnvironment],
            
            pop_size: int = 100, 
            gen_max: int = 10,
            
            selection_rate: float = 0.8,
            selection_decay: float = 1.0,
            selection_cap: float = 0.8,
            
            mutation_rate: float = 0.5,
            mutation_decay: float = 1.0,
            mutation_cap: float = 0.5,
            
            env_kwargs: dict = {},
        ):

        self.EnvClass = EnvClass
        self.env_kwargs = env_kwargs
        self.env = self.EnvClass(**self.env_kwargs)

        self.POPULATION_SIZE = pop_size
        self.GENERATION_MAX = gen_max

        self.selection_rate = selection_rate
        self.selection_decay = selection_decay
        self.selection_cap = selection_rate
        self.selection_greedy = max if self.selection_decay < 1.0 else min

        self.mutation_rate = mutation_rate
        self.mutation_decay = mutation_decay
        self.mutation_cap = mutation_rate
        self.mutation_greedy = max if self.mutation_decay < 1.0 else min

        self.solutions = [self.make_random_solutions() for _ in range(self.POPULATION_SIZE)]

        self.info = {
            'Generation': [],
            'Selection Rate': [],
            'Mutation Rate': [],
            'Scores': [],
            'Times': [],
            'Best Score': float('-inf')
        }

    def save(self, file_path: str):
        print(f"Saving {self.__class__.__name__} to '{file_path}'...", end='')

        with open(file_path, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done")

    @classmethod
    def load(cls, file_path: str) -> IAI:
        print(f"Loading {cls.__name__} from '{file_path}'...", end='')
            
        with open(file_path, 'rb') as file:
            ai = pickle.load(file)

        print('Done')

        return ai

    def make_random_solutions(self) -> np.ndarray:
        return np.random.uniform(-1, 1, size=(self.env.observation_size, self.env.action_size))


    def select_best(self, brains: list[np.ndarray], scores: list[float], brainsToSave : int):

        scores_normalized = (np.asarray(scores) - np.amin(np.asarray(scores))) / (np.amax(np.asarray(scores)) - np.amin(np.asarray(scores)))

        best_score = max(scores_normalized)
       
        # On trie brains et score_normalized de manière décroissante
        shuffled_list = list(zip(scores_normalized, brains))
        sortedList = sorted(zip(scores_normalized, brains), reverse=True, key=lambda x: x[0]) # list[tuple[float, brains]]
        new_brains = [brain for _, brain in sortedList[:2]]
        random.shuffle(shuffled_list)
        
        for score, brain in shuffled_list:
            if score/best_score > random.random():
                new_brains.append(brain)
               
            if len(new_brains) > brainsToSave:
                break

        return new_brains

    def reproductions(self, brains: list[np.ndarray]) -> list[np.ndarray]:

        def mutate(matrix):
            """
            Cette fonction effectue une mutation sur une matrice de valeurs comprises entre -1 et 1.

            Arguments :
            - matrix : une matrice numpy représentant une solution potentielle
            
            Retourne :
            - La matrice mutée
            """
            mutation_size = 0.01
            if np.random.rand() < self.mutation_rate:
                mutated_matrix: np.ndarray = matrix.copy()
                shape = matrix.shape
                mutation_indices = np.random.choice(np.prod(shape), int(self.mutation_rate * np.prod(shape)), replace=False)
                mutation_values = np.random.normal(0, 1, len(mutation_indices)) * mutation_size
                mutated_matrix.ravel()[mutation_indices] += mutation_values
                mutated_matrix = np.clip(mutated_matrix, -1, 1)
                return mutated_matrix
            else:
                return matrix

        def crossover(brain1: np.ndarray, brain2: np.ndarray) -> np.ndarray:

            split_index = random.randint(0, len(brain1)-1)
            new_brain = np.concatenate(
                (brain1[:split_index],
                brain2[split_index:])
                if random.random() < 0.5 else
                (brain2[split_index:],
                brain1[:split_index])
            )

            return new_brain

        new_brains = []
        while len(brains) + len(new_brains) < self.POPULATION_SIZE:
            new_brain = crossover(*random.sample(brains,k=2))
            new_brain = mutate(new_brain)
            new_brains.append(new_brain)
           
        return brains + new_brains

    def evolve(self, brains: list[np.ndarray], scores: list[float]) -> list[np.ndarray]:
        brains = self.select_best(brains, scores, int(self.selection_rate * self.POPULATION_SIZE))
        brains = self.reproductions(brains)
        
        return brains

    def train(self, show: bool = False, check_point: str = None):
        print("Training...")

        info = self.info.copy()
       
        best_score = float('-inf')

        for gen in range(1, self.GENERATION_MAX + 1):
            print(f"***** Generation {gen}/{self.GENERATION_MAX} *****")
            info['Generation'].append(gen)

            print(f"Taux Selection:", self.selection_rate)
            info['Selection Rate'].append(self.selection_rate)
            print(f"Taux Mutation:", self.mutation_rate)
            info['Mutation Rate'].append(self.mutation_rate)

            gen_time_start = time.time()
            
            scores = self.evaluate(self.solutions, show, (f"Generation {gen}/{self.GENERATION_MAX}"), best_score)
            info['Scores'].append(scores)
            
            best_score_gen = np.amax(scores)
            if best_score < best_score_gen:
                best_score = best_score_gen
                print("New best score:", best_score)

                if check_point is not None:
                    save_ai(self, check_point + f"_gen-{gen}-score-{best_score}")
            
            print("Best Score:", best_score_gen)
            print("Avg Score:", scores.mean())
            print("Std Score:", scores.std())

            print("Evolving...", end='')
            self.solutions = self.evolve(self.solutions, scores)
            print("Done")

            self.selection_rate = self.selection_greedy(self.selection_rate * self.selection_decay, self.selection_cap)
            self.mutation_rate = self.mutation_greedy(self.mutation_rate * self.mutation_decay, self.mutation_cap)

            info['Times'].append(time.time() - gen_time_start)
            print(f"Time: {info['Times'][-1]:0.2f} s")
            print()
        info['Best Score'] = best_score
        # l'ajoute à self à la fin car si check_point -> taille fichier énorme
        self.info = info

        print("Training done")

    def evaluate(self, solutions, show, gen, best_score) -> np.ndarray[float]:
        N_ENV = len(solutions)

        if show:
            pygame.display.init()
            pygame.font.init()
            screen = pygame.display.set_mode(self.EnvClass.display_size)
            clock = pygame.time.Clock()
            
            font=pygame.font.SysFont(None,  30)
            
            
            FPS = 60
        else:
            screen = None

        envs = VectorizedEnv(self.EnvClass, N_ENV, screen, **self.env_kwargs)

        obss, _ = envs.reset()
        dones = [False for _ in range(N_ENV)]
        total_scores = np.zeros(N_ENV)
        
        start_time = pygame.time.get_ticks() # enregistrement du temps actuel

        while not all(dones):
            actions = [self.get_action(obs, solution) for obs, solution in zip(obss, solutions)]

            obss, rewards, dones, _ = envs.step(actions)
            total_scores += np.asarray(rewards)

            if not show:
                continue

            for e in pygame.event.get():
                if e.type is pygame.QUIT:
                    exit(0)
                    
            # Mise à jour du temps écoulé
            elapsed_time = pygame.time.get_ticks() - start_time
            seconds = int(elapsed_time / 1000)
            seconds_str = str(seconds)
            time_text = font.render("Temps génération actuelle: " + seconds_str, True, (0,0,0))
            
            gen_text = font.render(gen, True, (0,0,0))
            
            bestScore_text = font.render("Meilleur score gen actuelle: ???", True, (0,0,0))
            bestScoreGlob_text = font.render("Meilleur score global: ???", True, (0,0,0))
            
            tauxSelec_text = font.render(f"Taux Selection: {self.selection_rate:.4f}", True, (0,0,0))
            tauxMut_text = font.render(f"Taux Mutation: {self.mutation_rate:.4f}", True, (0,0,0))
            
            # screen.fill("white")
            envs.render(range(show))
            screen.blit(time_text, (10,10))
            screen.blit(gen_text, (10, 35))
            screen.blit(bestScore_text, (10, 60))
            screen.blit(bestScoreGlob_text, (10, 85))
            screen.blit(tauxSelec_text, (10, 110))
            screen.blit(tauxMut_text, (10, 135))
            
            
            pygame.display.flip()
            clock.tick(FPS)
            pygame.display.set_caption(str(clock.get_fps()))

        return total_scores

    def test(self, show: bool = True, batch_size: int = 10):
        print("Testing...")

        scores = []
        for idx in range(0, len(self.solutions)-1, batch_size):
            print(loadingBar(idx, len(self.solutions)), end='')
            scores.extend(self.evaluate(self.solutions[idx:idx+batch_size], show))
        print(loadingBar(len(self.solutions), len(self.solutions)), end='')    
        print("Best score:", max(scores))
        print("Avg score:", stats.fmean(scores))
        
        print("Testing...Done")

    @staticmethod
    def get_action(obs: np.ndarray, policy: np.ndarray):
        act = np.dot(obs, policy)
        act= (act%2) - 1  # "normalise" entre -1 et 1
        return act


class GeneticAlgoByTime(GeneticAlgo):

    def __init__(
            self, 
            EnvClass: Type[IEnvironment] = WalkingEnv,  
            pop_size: int = 100, 
            gen_max: int = 10, 
            time_max: int = 1000,
            **kwargs
        ):
        self.time_max = time_max
        
        super().__init__(EnvClass, pop_size, gen_max, env_kwargs={'time_max': self.time_max}, **kwargs)

    def make_random_solutions(self) -> np.ndarray:
        return np.random.uniform(-1, 1, size=(self.time_max, self.env.action_size))

    def get_action(self, obs: np.ndarray, policy: np.ndarray) -> np.ndarray:
        return policy[int(obs[-1]) % self.time_max]
 
