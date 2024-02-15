import pygame


class Window:

    def __init__(self, dims: tuple[int, int] = (600, 400), fps: int = 60, caption: str = "fps: %fps%"):
        self.dims = dims
        self.FPS = fps
        self.fps = self.FPS
        self.caption = caption

        pygame.init()
        self.screen = pygame.display.set_mode(self.dims, flags=pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)

    def add_text(self, text: str, pos: tuple[int, int] = (10, 10), color: tuple[int, int, int] = (0, 0, 0)):
        self.screen.blit(self.font.render(text, True, color), pos)

    def render(self):
        self.process_event()
        pygame.display.flip()
        self.clock.tick(self.fps)
        pygame.display.set_caption(self.caption.replace("%fps%", f"{self.clock.get_fps():.0f}"))

    def process_event(self):

        # self.add_text("Press f to switch between uncapped and capped fps", (10, self.dims[1] - 40))

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                exit(0)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_f: # unkap fps
                self.fps = 0 if self.fps == self.FPS else self.FPS

        
