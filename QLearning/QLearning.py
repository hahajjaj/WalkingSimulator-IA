from math import floor
import pygame
from typing import Type
import pygame
from environment import IEnvironment, WalkingEnv
from ia import IAI
from environment import WalkingEnv
import random
import numpy as np
import pickle
import logging
import sys

class QLearning(IAI):
    def __init__(self, EnvClass: Type[IEnvironment], epsilon=0.5,gamma=0.8,alpha=0.2, numTraining=sys.maxsize, stepToSave=5000000) -> None:
        '''
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = gamma
        self.numTraining = numTraining

        assert EnvClass == WalkingEnv, "Only works with WalkingEnv" 
        self.EnvClass = EnvClass
        
        self.stepToSave = stepToSave
        self.weight = []
        self.Qvalues = {}
        # logging.basicConfig(level=logging.DEBUG)
    
    def getQValue(self, state, action):
        return self.Qvalues.setdefault(state, {action:random.random()-0.5 * 10 for action in self.getLegalActions(state)})[action]
    
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        return max([self.getQValue(state, action) for action  in self.getLegalActions(state)], default=0.0)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        return max([(self.getQValue(state, action), action) for action  in self.getLegalActions(state)], default=(0.0, None))[1]



    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # TODO
        if random.random()>self.epsilon:
          return random.choice([action for action in self.getLegalActions(state)])
        else:
          return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        '''
        update the qV of one state action with the knowledge of the reward
        '''
        self.Qvalues[state][action] = (1-self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


    def getLegalActions(self, state):
        actions = [(i, j) for i in range(-self.env.action_size+1, self.env.action_size) for j in range(-self.env.action_size+1, self.env.action_size)]
        return actions

    def toNpAction(self, action):
        retAction = np.zeros(self.env.action_size)
        neg = False
        action = list(action)
        if action[0] < 0:
            neg = True
            action[0] *= -1
        retAction[action[0]] = 1 * (-1 if neg else 1)
        neg = False
        if action[1] < 0:
            neg = True
            action[1] *= -1
        retAction[action[1]] = 1 * (-1 if neg else 1)
        return retAction

    def getState(self):
        # return tuple([floor(b.angle) for b in self.env.bodies])
        torso = self.env.chassis_body.position
        state = []
        for b in self.env.bodies:
            s = [1,1]
            if b.position[0] < torso[0]:
                s[0] = -1
            if b.position[1] < torso[1]:
                s[1] = -1
            state.extend(s)
        return tuple(state)

    def train(self, show: bool = False):

        if show:
            pygame.init()
            screen = pygame.display.set_mode(self.EnvClass.display_size)
            font=pygame.font.SysFont(None,  30)
            clock = pygame.time.Clock()
            FPS = self.EnvClass.render_fps
        else:
            screen = None

            
        self.env = self.EnvClass(screen=screen)
        self.obs, _ = self.env.reset()
        lastState = self.getState()
        episode = 0
        while episode < 200:
            done = False
            action = self.getAction(lastState)
            actionNp = self.toNpAction(action)
            logging.debug(lastState)
            self.obs, reward, _, _ = self.env.step(actionNp)
            state = self.getState()
            # reward += self.env.chassis_body.position[0]
            reward *= 10
            if self.env.chassis_body.position[0] > 1080:
                reward = 1000
                done = True
            if self.env.time_step > 36000:
                reward = -100
                done = True

            self.update(lastState, action, state, reward)
            print(f"e:{episode} r: {reward},t: {self.env.time_step},position:{self.env.chassis_body.position[0]}")
            if done:
                episode += 1
                if episode % 10 == 0:
                    self.epsilon = 0.9
                else:
                    self.epsilon = 0.5
                    # self.save(f"QL-E:3.{episode}-score:{reward:.2f}-tps:{self.env.time_step}-position:{self.env.chassis_body.position[0]}")
                self.env.reset()
            
            lastState = state

            if not show: continue

            for e in pygame.event.get():
                if e.type is pygame.QUIT:
                    exit(0)
            
            self.env.render()

            screen.blit(font.render(f"alpha: {self.alpha}, episode: {episode}, step: {self.env.time_step} epsilone: {self.epsilon}, reward: {reward:.3f}", True, (0, 0, 0)), (10, 10))
            pygame.display.flip()
            clock.tick(FPS)
            pygame.display.set_caption(f"{self.__class__.__name__} (fps: {clock.get_fps():.0f})")


    def test(self):
        pass

    def save(self, saveFile: str):
        with open(saveFile, 'wb') as file:
            pickle.dump(self.Qvalues, file, protocol=pickle.HIGHEST_PROTOCOL)

    def fromFile(self, saveFile: str):
        with open(saveFile, 'rb') as file:
            self.Qvalues = pickle.load(file)


if __name__ == "__main__":
    ai = QLearning(WalkingEnv)
    # ai.fromFile("/home/stan/Documents/ULB/proj_anne/PROJET-BA3/QL-E:200-score:-1000.00-tps:36001-position:230.6650483935575")
    ai.train(show=True)
    # logging.warning("input")
    # input()
    # ai.epsilon = 0.75
    # ai.train(show=True)
    # ai.save()