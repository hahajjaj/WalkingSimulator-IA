
from typing import Iterable, Optional, Protocol, Type

import numpy as np
import pygame

import pymunk
import pymunk.pygame_util
from pymunk import Vec2d


class IEnvironment(Protocol):

    def __init__(self, screen: Optional[pygame.Surface] = None, **kwargs):
        ...

    @staticmethod
    @property
    def observation_size(self) -> int:
        ...

    @staticmethod
    @property
    def action_size(self) -> int:
        ...

    @staticmethod
    @property
    def display_size(self) -> tuple[int, int]:
        ...

    @staticmethod
    @property
    def render_fps(self) -> int:
        ...

    def reset(self) -> tuple[np.ndarray, dict]:
        ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        ...

    def render(self, _bg: bool = True):
        ...

class VectorizedEnv:

    observation_shape: tuple[int, int]
    action_shape: tuple[int, int]
    observation_size: int
    action_size: int
    display_size: tuple[int, int]
    render_fps: int

    def __init__(self, EnvClass: Type[IEnvironment], n_env: int, screen: Optional[pygame.Surface] = None, **kwargs):
        self.EnvClass = EnvClass
        self.N_ENV = n_env
        self.screen = screen

        self.envs = [self.EnvClass(screen=self.screen, **kwargs) for _ in range(self.N_ENV)]

        self.observation_size = self.envs[0].observation_size
        self.observation_shape = (self.N_ENV, self.observation_size)
        self.action_shape = (self.N_ENV, self.envs[0].action_size)
        self.display_size = self.EnvClass.display_size
        self.render_fps = self.EnvClass.render_fps


    def reset(self) -> tuple[tuple[np.ndarray], tuple[dict]]:
        return zip(*[env.reset() for env in self.envs])
    
    def step(self, actions: Iterable[np.ndarray]) -> tuple[tuple[np.ndarray], tuple[float], tuple[bool], tuple[dict]]:
        return zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        
    def render(self, indexs: Optional[Iterable[int]] = None, _bg: bool = True):
        if self.screen is None:
            return
                
        if indexs is None:
            indexs = range(self.N_ENV)
        first = indexs[0]
        for idx in indexs:
            self.envs[idx].render(_bg and idx == first)


class WalkingEnv:

    observation_size: int = 4
    action_size: int = 4
    display_size: tuple[int, int] = (1200, 600)
    render_fps: int = 60

    GROUND_Y: int = 550
    
    def __init__(self, time_max: int = 1000, sim_step: float = 0.02, screen: Optional[pygame.Surface] = None):
        self.info = {}

        self.time_max = time_max
        self.time_step = 0

        self.SIMULATION_STEP = sim_step

        self.space = None

        self.ground = None

        self.bodies = []
        self.shapes = []
        self.constraints = []

        self.screen = screen
        if isinstance(self.screen, pygame.Surface):
            self.buff_screen = pygame.Surface(self.screen.get_rect().size, pygame.SRCALPHA)
            self.draw_options = pymunk.pygame_util.DrawOptions(self.buff_screen)
            self.background_image = pygame.image.load("fild.jpeg")

    def _reset(self):
        CHASSIS_DIMS: tuple[float, float] = (50.0, 50.0)
        CHASSIS_INITIAL_POS: Vec2d = Vec2d(CHASSIS_DIMS[0] + 150.0, self.GROUND_Y - CHASSIS_DIMS[1] - 10)
        CHASSIS_MASS: float = 10.0

        LEG_DIMS: tuple[float, float] = (50.0, 10.0)
        LEG_MASS: float = 1.0

        RELATIVE_ANGU_VEL: float = 0.0

        self.motors = []

        self.chassis_body = pymunk.Body(
            CHASSIS_MASS, pymunk.moment_for_box(CHASSIS_MASS, CHASSIS_DIMS))
        self.chassis_body.position = CHASSIS_INITIAL_POS
        self.chassis_shape = pymunk.Poly.create_box(
            self.chassis_body, CHASSIS_DIMS, 0.5)
        self.chassis_shape.color = pygame.Color(200, 200, 200)
        self.bodies.append(self.chassis_body)
        self.shapes.append(self.chassis_shape)


        leftLeg_a_body = pymunk.Body(
            LEG_MASS, pymunk.moment_for_box(LEG_MASS, LEG_DIMS))
        leftLeg_a_body.position = CHASSIS_INITIAL_POS - ( (CHASSIS_DIMS[0] / 2.0) + (LEG_DIMS[0] / 2.0), 0 )
        leftLeg_a_shape = pymunk.Poly.create_box(
            leftLeg_a_body, LEG_DIMS, 0.1)
        leftLeg_a_shape.color = pygame.Color(255, 0, 0)
        self.bodies.append(leftLeg_a_body)
        self.shapes.append(leftLeg_a_shape)

        leftLeg_b_body = pymunk.Body(
            LEG_MASS, pymunk.moment_for_box(LEG_MASS, LEG_DIMS))
        leftLeg_b_body.position = leftLeg_a_body.position - ( LEG_DIMS[0], 0 )
        leftLeg_b_shape = pymunk.Poly.create_box(
            leftLeg_b_body, LEG_DIMS, 0.1)
        leftLeg_b_shape.color = pygame.Color(0, 255, 0)
        self.bodies.append(leftLeg_b_body)
        self.shapes.append(leftLeg_b_shape)

        rightLeg_a_body = pymunk.Body(
            LEG_MASS, pymunk.moment_for_box(LEG_MASS, LEG_DIMS))
        rightLeg_a_body.position = CHASSIS_INITIAL_POS + ( (CHASSIS_DIMS[0] / 2.0) + (LEG_DIMS[0] / 2.0), 0 )
        rightLeg_a_shape = pymunk.Poly.create_box(
            rightLeg_a_body, LEG_DIMS, 0.1)
        rightLeg_a_shape.color = pygame.Color(255, 0, 0)
        self.bodies.append(rightLeg_a_body)
        self.shapes.append(rightLeg_a_shape)

        rightLeg_b_body = pymunk.Body(
            LEG_MASS, pymunk.moment_for_box(LEG_MASS, LEG_DIMS))
        rightLeg_b_body.position = rightLeg_a_body.position + (LEG_DIMS[0], 0)
        rightLeg_b_shape = pymunk.Poly.create_box(
            rightLeg_b_body, LEG_DIMS, 0.1)
        rightLeg_b_shape.color = pygame.Color(0, 255, 0)
        self.bodies.append(rightLeg_b_body)
        self.shapes.append(rightLeg_b_shape)
        
        joint_left_ab = pymunk.PinJoint(
            leftLeg_b_body, leftLeg_a_body, (LEG_DIMS[0]  / 2.0, 0), (-LEG_DIMS[0] / 2.0, 0))
        motor_left_ab = pymunk.SimpleMotor(
            leftLeg_b_body, leftLeg_a_body, RELATIVE_ANGU_VEL)
        self.constraints.extend([joint_left_ab, motor_left_ab])
        self.motors.append(motor_left_ab)

        joint_left_ac = pymunk.PinJoint(
            leftLeg_a_body, self.chassis_body, (LEG_DIMS[0] / 2.0, 0), (-CHASSIS_DIMS[0] / 2.0, 0))
        motor_left_ac = pymunk.SimpleMotor(
            leftLeg_a_body, self.chassis_body, RELATIVE_ANGU_VEL)
        self.constraints.extend([joint_left_ac, motor_left_ac])
        self.motors.append(motor_left_ac)

        joint_right_ab = pymunk.PinJoint(
            rightLeg_b_body, rightLeg_a_body, (-LEG_DIMS[0] / 2.0, 0), (LEG_DIMS[0] / 2.0, 0))
        motor_right_ab = pymunk.SimpleMotor(
            rightLeg_b_body, rightLeg_a_body, RELATIVE_ANGU_VEL)
        self.constraints.extend([joint_right_ab, motor_right_ab])
        self.motors.append(motor_right_ab)

        joint_right_ac = pymunk.PinJoint(
            rightLeg_a_body, self.chassis_body, (-LEG_DIMS[0] / 2.0, 0), (CHASSIS_DIMS[0] / 2.0, 0))
        motor_right_ac = pymunk.SimpleMotor(
            rightLeg_a_body, self.chassis_body, RELATIVE_ANGU_VEL)
        self.constraints.extend([joint_right_ac, motor_right_ac])
        self.motors.append(motor_right_ac)

        self.old_pos = self._get_position()
            
        shape_filter = pymunk.ShapeFilter(group=1)
        for shape in self.shapes:
            shape.filter = shape_filter
            shape.friction = 0.9

    def move_ground(self, center: Vec2d):
        if self.ground is not None:
            self.space.remove(self.ground)

        assert center.y == self.GROUND_Y, ValueError("Ground y coordinate cannot change")
        
        self.ground = pymunk.Segment(
            self.space.static_body, center - self.ground_normal / 2.0, center + self.ground_normal / 2.0, 20.0)
        self.ground.friction = 0.6
        self.ground.color = pygame.Color(127, 172, 113, 255)
        self.ground_center = center
        self.space.add(self.ground)
    
    def reset(self) -> tuple[np.ndarray, dict]:
        self.bodies.clear()
        self.shapes.clear()
        self.constraints.clear()

        self.time_step = 0
        
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 98.1)

        self._reset()

        self.initial_pos = self._get_position()
        
        self.ground = None
        self.ground_normal = Vec2d(self.display_size[0] + 1000.0, 0.0)
        self.move_ground(Vec2d(self.display_size[0] / 2.0, self.GROUND_Y))

        self.space.add(*self.bodies)
        self.space.add(*self.shapes)
        self.space.add(*self.constraints)

        return (self.get_obs(), self.info)
    
    def _step(self, action: np.ndarray):
        for motor, act in zip(self.motors, action):
            motor.rate = act

    def get_obs(self) -> np.ndarray:
        return np.array([*self._get_position(), self.time_step])
    
    def get_reward(self) -> float:
        reward = 0.0

        new_pos = self._get_position()
        reward += new_pos.x - self.old_pos.x
        self.old_pos = new_pos
        
        if not self.chassis_shape.shapes_collide(self.ground).points:
            reward += 50/self.time_max

        return reward

    def get_done(self) -> bool:
        return self.time_step > self.time_max

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        '''
        return: obs, reward, done, info
        '''
        assert self.space is not None, RuntimeError("Call reset before rendering the environment")


        self._step(action)

        self.space.step(self.SIMULATION_STEP)
        
        # move ground to always be below animal
        animal_position = self._get_position()
        if abs(self.ground_center.x - animal_position.x) > 50.0:
            self.move_ground(Vec2d(animal_position.x, self.GROUND_Y))
        res = (
            self.get_obs(),
            self.get_reward(),
            self.get_done(),
            self.info
        )
        self.time_step += 1
        return res
    
    def _get_position(self) -> Vec2d:
        return self.chassis_body.position
    
    def render(self, _bg: bool = True):
        assert self.space is not None, RuntimeError("Call reset before rendering the environment")
        
        if self.screen is None:
            return
        
        # if _bg:

            # self.buff_screen.fill("black")
        if _bg:
            delta = tuple(map(int, (self.initial_pos + (60, 0) - self._get_position())[:2]))
            self.draw_options.transform = pymunk.Transform.translation(delta[0], 0)
            self.buff_screen.blit(self.background_image, (delta[0]-60, 0))
            self.buff_screen.blit(self.background_image, (delta[0]-60+1200, 0))
            self.buff_screen.blit(self.background_image, (delta[0]-60+2400, 0))
            

        self.space.debug_draw(self.draw_options)
        self.screen.blit(self.buff_screen, (0, 0))


class Segment:

    def __init__(self, mass: float, a: Vec2d | tuple[int, int], b: Vec2d | tuple[int, int], radius: float, old=False):
        a = a if isinstance(a, Vec2d) else Vec2d(*a)
        b = b if isinstance(b, Vec2d) else Vec2d(*b)
        self._normal = b - a if not old else b
        self.length = self._normal.length
        self.body = pymunk.Body(mass, pymunk.moment_for_segment(mass, (0, 0), self._normal, radius))
        self.body.position = a
        self.shape = pymunk.Segment(self.body, (0, 0), self._normal, radius)

    @property
    def a(self) -> Vec2d:
        return self.position
    @property
    def b(self) -> Vec2d:
        return self.position + self.normal
    @property
    def position(self) -> Vec2d:
        return self.body.position
    @property
    def center(self) -> Vec2d:
        return self.position + self.normal / 2
    @property
    def normal(self) -> Vec2d:
        return self._normal.rotated_degrees(self.body.angle)
    
    @property
    def angle(self) -> float:
        return self.body.angle
    @property
    def angular_velocity(self) -> float:
        return self.body.angular_velocity
    @property
    def torque(self) -> float:
        return self.body.torque
    @property
    def velocity(self) -> Vec2d:
        return self.body.velocity
        

class Muscle(pymunk.constraints.DampedSpring):

    def __init__(self, a: pymunk.Body, b: pymunk.Body, anchor_a: Vec2d, anchor_b: Vec2d, max_force: float = 100):
        self.initial_rest_length = (a.position + anchor_a).get_distance(b.position + anchor_b)
        super().__init__(a, b, anchor_a, anchor_b, self.initial_rest_length, max_force, max_force * 0.05)

        self.max_force = max_force

        self._contraction = 0.0
    
    @property
    def contraction(self) -> float:
        return self._contraction
    
    @contraction.setter
    def contraction(self, contraction: float) -> float:
        self._contraction = contraction

        self.rest_length = self.initial_rest_length + self._contraction * self.initial_rest_length * 0.75

        return self._contraction
            

class ConfigurableAnimalEnv(WalkingEnv):

    observation_size: int = None
    action_size: int = None
    display_size: tuple[int, int] = (1200, 600)
    render_fps: int = 60

    _DEFAULT_CONFIG: dict = {
        'bones': [ # tuple de positions des extrémité des bones
            ((0, 0), (9, 0)),
            ((0, 0), (2, 3)), 
            ((2, 3), (0, 5)),
            ((0, 5), (3, 5)),
            ((9, 0), (8, 2)),
            ((8, 2), (11, 2))
        
        ],
        'joints': [ # indexs des bonnes relié par un joint plus index du point de pivot pour le bone a (ici a=0, b=1, pp=(0, 0))
            (0, 1, 0),
            (1, 2, 1),
            (2, 3, 1),
            (0, 4, 1),
            (4, 5, 1)
        ],
        'muscles': [ # index des bones relié par un muscle
            (0, 1),
            (0, 3),
            (1, 2),
            (2, 3),
            (0, 4),
            (0, 5),
            (4, 5)
        ],
        'bone_mass': 1.0,
        'bone_radius': 5.0,
        'bone_friction': 0.5,
        'muscle_max_force': 100.0,
        'scale': 20.0
    }

    # grenouille
    _FROG_CONFIG: dict = {
        'bones': [
            ((0, 0), (-5, -1)),
            ((-5, -1), (2, -3)),
            ((2, -3), (-1, -5)),
            ((-1, -5), (5, -7)),
            ((5, -7), (4, -4)),
            ((4, -4), (8, -1)),
        ],
        'joints': [
            (0, 1, 1),
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
        ],
        'muscles': [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
    }

    _KANGAROO_CONFIG: dict = {
        'bones': [
            ((0, 0), (-6, -2)),
            ((0, 0), (-7, -5)),
            ((-6, -2), (-7, -5)),
            ((-7, -5), (-7, -10)),
            ((-7, -5), (2, -13)),
            ((-7, -10), (2, -13)),
            ((2, -13), (1, -17)),
            ((2, -13), (6, -16)),
        ],
        'joints': [
            (0, 1, 1),
            (0, 2, 0),
            (1, 2, 1),
            (1, 3, 1),
            (1, 4, 1),
            (3, 5, 1),
            (4, 5, 1),
            (4, 6, 1),
            (4, 7, 1),
        ],
        'muscles': [
            (0, 3),
            (0, 5),
            (1, 4),
            (1, 5),
            (2, 4),
            (3, 6),
            (3, 7)
        ]
    }

    def __init__(self, config_animal: dict = _DEFAULT_CONFIG,
                  time_max: int = 1000, sim_step: float = 0.02, screen: Optional[pygame.Surface] = None):
        super().__init__(time_max, sim_step, screen)

        assert 'joints' in config_animal and 'bones' in config_animal and 'muscles' in config_animal and len(config_animal['bones']) >= 1, ValueError(f"Wrong format of config_animal")
        self.config_animal = config_animal

        self.observation_size = len(self.config_animal['joints']) * 7 + 1 # position, angle, vélocité angulaire, torque/couple et vélocité de chaque bones + time_step
        self.action_size = len(self.config_animal['muscles']) # une action par muscles

        if self.screen:
            self.draw_options.shape_dynamic_color = (255, 0, 0, 255)

    def _reset(self):
        self.space.gravity = (0.0, 200.0)
        
        joints = self.config_animal['joints']
        bones = self.config_animal['bones']
        muscles = self.config_animal['muscles']

    
        bone_mass = self.config_animal.setdefault('bone_mass', 1.0)
        bone_radius = self.config_animal.setdefault('bone_radius', 5.0)
        bone_friction = self.config_animal.setdefault('bone_friction', 0.9)

        muscle_max_force = self.config_animal.setdefault('muscle_max_force', 100.0)

        scale = self.config_animal.setdefault('scale', 20.0)

        self.joints = []
        self.bones = []
        self.muscles = []


        for pos_a, pos_b in bones:
            bone = Segment(bone_mass, Vec2d(*pos_a) * scale, Vec2d(*pos_b) * scale, bone_radius)
            bone.shape.friction = bone_friction
            self.bones.append(bone)
            self.bodies.append(bone.body)
            self.shapes.append(bone.shape)
        
        min_xy = min([bone.a.x if bone.a.x < bone.b.x else bone.b.x for bone in self.bones]), min([bone.a.y if bone.a.y < bone.b.y else bone.b.y for bone in self.bones])
        max_xy = max([bone.a.x if bone.a.x > bone.b.x else bone.b.x for bone in self.bones]), max([bone.a.y if bone.a.y > bone.b.y else bone.b.y for bone in self.bones])
        translation = Vec2d(20, self.GROUND_Y - 50) - (min_xy[0], max_xy[1])
        for bone in self.bones:
            bone.body.position += translation

        center = Vec2d((max_xy[0] - min_xy[0] / 2), (max_xy[1] - min_xy[1]) / 2)
        self.offset = center - self.bones[0].position

        for a, b, pp in joints:
            joint = pymunk.constraints.PivotJoint(self.bones[a].body, self.bones[b].body, 
                                                  self.bones[a].a if pp == 0 else self.bones[a].b)
            self.joints.append(joint)
            self.constraints.append(joint)

        for a, b in muscles:
            muscle = Muscle(self.bones[a].body, self.bones[b].body, 
                            self.bones[a].normal / 2, self.bones[b].normal / 2,
                            muscle_max_force)
            self.muscles.append(muscle)
            self.constraints.append(muscle)

        shape_filter = pymunk.ShapeFilter(group=1)
        for shape in self.shapes:
            shape.filter = shape_filter

        self.old_x = self._get_min_x()

    def _get_min_x(self) -> int:
        return min([bone.a.x if bone.a.x < bone.b.x else bone.b.x for bone in self.bones])
    
    def get_obs(self) -> np.ndarray:
        return np.array([
            [*bone.position, bone.angle, bone.angular_velocity, bone.torque, *bone.velocity]
             for bone in self.bones
        ]).flatten()
    
    def get_reward(self) -> float:
        
        reward = 0

        new_x = self._get_min_x()
        reward += new_x - self.old_x
        self.old_x = new_x

        return reward
    
    def _step(self, actions: np.ndarray):
        for muscle, action in zip(self.muscles, actions):
            muscle.contraction = action

    def _get_position(self) -> Vec2d:
        return self.bones[0].position + self.offset



    


if __name__ == '__main__':
    # Exemple d'utilisation d'un environment
    EnvClass = ConfigurableAnimalEnv

    # si on montre l'environnement
    pygame.init()
    screen = pygame.display.set_mode(EnvClass.display_size)
    clock = pygame.time.Clock()
    FPS = EnvClass.render_fps

    env = EnvClass(screen=screen) # screen peut être None -> pas de rendu (show == False)

    obs, _ = env.reset()
    done = False

    while not done:

        obs, score, done, _ = env.step(np.random.uniform(-1, 1, env.action_size))
        
        for e in pygame.event.get():
            if e.type is pygame.QUIT:
                exit(0)

        # dessine les environnements sur le screen donné à la création de l'environnement, si None -> ne fait rien
        # peut décider de dessiner le background avec l'argument 'bg'
        env.render()
        # dessine info additionel si il faut (ex: gui)
        pygame.display.flip()
        clock.tick(FPS)
        pygame.display.set_caption(f"Exemple simple {EnvClass.__name__} (fps: {clock.get_fps():.0f})")


if __name__ == '__main__':
    # Exemple d'utilisation d'un environnement vectorisé, cad plusieur fois le même environnement
    EnvClass = ConfigurableAnimalEnv

    # si on montre les environnements
    pygame.init()
    screen = pygame.display.set_mode(EnvClass.display_size)
    clock = pygame.time.Clock()
    FPS = EnvClass.render_fps
    
    N_ENV = 10
    envs = VectorizedEnv(EnvClass, N_ENV, screen) # screen peut être None -> pas de rendu (show == False)

    obss, _ = envs.reset()
    dones = [False for _ in range(N_ENV)]

    while not all(dones):

        obss, scores, dones, _ = envs.step(np.random.uniform(-1, 1, envs.action_shape))
        
        for e in pygame.event.get():
            if e.type is pygame.QUIT:
                exit(0)
        
        # dessine les environnements sur le screen donné à la création de l'environnement, si None -> ne fait rien
        # peut décider de dessiner le background avec l'argument 'bg'
        # peut donné une list d'indexs des environments à dessiné dans l'ordre donné, par defaut tous dans l'ordre de creation
        envs.render(range(N_ENV))
        # dessine info additionel si il faut (ex: gui)
        pygame.display.flip()
        clock.tick(FPS)
        pygame.display.set_caption(f"Exemple Vectorized {EnvClass.__name__} (fps: {clock.get_fps():.0f})")
