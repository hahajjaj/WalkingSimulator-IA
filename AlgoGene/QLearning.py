import pygame
from ia import IAI
import random
import numpy as np
import pickle
import logging
class QLearning(IAI):
    def __init__(self, EnvClass, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=1000
                 , episode=100, env_kwargs: dict = {}) -> None:
        '''
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        '''
        self.EnvClass = EnvClass
        self.episode = episode
        self.env_kwargs = env_kwargs

        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = gamma
        self.numTraining = numTraining
        
        self.weight = []
        self.Qvalues = {}
        # logging.basicConfig(level=logging.DEBUG)
    
    def getQValue(self, state, action):
        return self.Qvalues.setdefault(state, {action:0.0 for action in self.getLegalActions(state)})[action]
    
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
        if random.random()>0.5:
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
        actions = [i for i in range(-self.env.action_size+1, self.env.action_size)]
        return actions

    def toNpAction(self, action):
        neg = False
        retAction = np.zeros(self.env.action_size)
        if action < 0:
            neg = True
            action *= -1
        retAction[action] = 1 * (-1 if neg else 1)
        return retAction

    def getState(self):
        return tuple([b.angle for b in self.env.bodies])

    def train(self, show: bool = False):
        if show:
            pygame.init()
            screen = pygame.display.set_mode(self.EnvClass.display_size)
            clock = pygame.time.Clock()
            FPS = self.EnvClass.render_fps
        else:
            screen = None

        best_score = float('-inf')
        self.env = self.EnvClass(screen=screen, **self.env_kwargs)
        for ep in range(self.episode):
            print("Episode:", ep)

            self.obs, _ = self.env.reset()

            lastState = self.getState()
            done = False
            score = 0.0
            while not done:
                action = self.getAction(lastState)
                actionNp = self.toNpAction(action)
                # logging.debug(lastState)
                self.obs, reward, done, _ = self.env.step(actionNp)
                state = self.getState()
                self.update(lastState, action, state, reward)
                score += reward
                lastState = state

                if not show:
                    continue

                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        exit(0)
                self.env.render()
                pygame.display.flip()
                clock.tick(FPS)

            if score > best_score:
                best_score = score
                print("New best score:", best_score)

    def test(self):
        pass

    def save(self, saveFile: str):
        with open(saveFile, 'wb') as file:
            pickle.dump(self.Qvalues, file, protocol=pickle.HIGHEST_PROTOCOL)

    def fromFile(self, saveFile: str):
        self.Qvalues = pickle.load(saveFile)


if __name__ == "__main__":
    ai = QLearning()
    # ai.fromFile()
    ai.train(show=True)
    # ai.save()