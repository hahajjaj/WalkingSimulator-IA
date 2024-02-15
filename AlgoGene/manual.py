from environment import WalkingEnv
import pygame
import numpy as np
class Manual:
    def __init__(self, screen: pygame.Surface):


        self.env = WalkingEnv(screen=screen)
        self.clock = pygame.time.Clock()
    
    def train(self):
        self.env.reset()
        running = True
        keyUp = [pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r, pygame.K_t, pygame.K_y, pygame.K_u, pygame.K_i, pygame.K_o, pygame.K_p]
        keyDown = [pygame.K_a,pygame.K_s, pygame.K_d, pygame.K_f, pygame.K_g, pygame.K_h, pygame.K_j, pygame.K_k, pygame.K_l]
        actionRepeatUp = [False for _ in keyUp]
        actionRepeatDown = [False for _ in keyDown]
        while running:
            action = np.zeros((self.env.action_size,1))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    pygame.image.save(self.env.screen, "balls_and_lines.png")
                for i, k in enumerate(keyUp):
                    if event.type == pygame.KEYDOWN and event.key == k:
                        actionRepeatUp[i] = True
                    elif event.type == pygame.KEYUP and event.key == k:
                        actionRepeatUp[i] = False
                for i, k in enumerate(keyDown):
                    if event.type == pygame.KEYDOWN and event.key == k:
                        actionRepeatDown[i] = True
                    elif event.type == pygame.KEYUP and event.key == k:
                        actionRepeatDown[i] = False

            for a, r in enumerate(actionRepeatUp):
                if r:
                    action[a] += 0.2
                if a >= len(action)-1:
                    break
            for a, r in enumerate(actionRepeatDown):
                if r:
                    action[a] -= 0.2
                if a >= len(action)-1:
                    break
            self.env.step(action)
            self.env.render()
            pygame.display.flip()
            self.clock.tick(self.env.render_fps)
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode(WalkingEnv.display_size)
    Manual(screen).train()
    pygame.display.quit()