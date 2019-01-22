import math
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from examples.levi.input_formatter import LeviInputFormatter
from examples.levi.output_formatter import LeviOutputFormatter
import time
import sys
import os


class PythonExample(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this is for separate process imports
        from examples.levi.torch_model import SymmetricModel
        self.Model = SymmetricModel
        self.torch = torch
        self.controller_state = SimpleControllerState()
        self.frame = 0  # frame counter for timed reset
        self.brain = 0  # bot counter for generation reset
        self.pop = 10  # population for bot looping
        self.num_best = 2
        self.gen = 0
        self.min_distance_to_ball = []
        self.ball_set = ((0, 1000, 0), (0, 0, 0)), ((1000, 0, 1000), (0, 0, 0)), \
                        ((-1000, 0, 1000), (0, 0, 0)), ((-2000, -2000, 1000), (0, 0, 0)), \
                        ((2000, -2000, 1000), (0, 0, 0))
        self.attempt = 0
        self.max_frames = 5000
        self.bot_list = [self.Model() for _ in range(self.pop)]  # list of Individual() objects
        self.bot_list[-self.num_best:] = [self.Model()] * self.num_best  # make sure last bots are the same
        self.bot_fitness = [0] * self.pop
        self.fittest = None  # fittest object
        self.mut_rate = 2  # mutation rate
        self.distance_to_ball = [math.inf] * self.max_frames  # set high for easy minimum
        self.input_formatter = LeviInputFormatter(team, index)
        self.output_formatter = LeviOutputFormatter(index)

    def initialize_agent(self):
        self.reset()  # reset at start

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # NEURAL NET INPUTS
        inputs = self.input_formatter.create_input_array([packet])
        inputs = [self.torch.from_numpy(x).float() for x in inputs]

        my_car = packet.game_cars[self.index]
        distance_to_ball_x = packet.game_ball.physics.location.x - my_car.physics.location.x
        distance_to_ball_y = packet.game_ball.physics.location.y - my_car.physics.location.y
        distance_to_ball_z = packet.game_ball.physics.location.z - my_car.physics.location.z
        self.distance_to_ball[self.frame] = math.sqrt(
            distance_to_ball_x ** 2 + distance_to_ball_y ** 2 + distance_to_ball_z ** 2)

        # RENDER RESULTS
        action_display = "GEN: " + str(self.gen + 1) + " | BOT: " + str(self.brain + 1)

        draw_debug(self.renderer, action_display)

        # STOP EVOLVING WHEN THE BALL IS TOUCHED
        #if packet.game_ball.latest_touch.player_name == "Self-Evolving-Car":
        #  self.mut_rate = 0

        # GAME STATE
        car_state = CarState(boost_amount=100)
        velocity = self.ball_set[self.attempt][1]
        ball_state = BallState(
            Physics(velocity=Vector3(velocity[0], velocity[1], velocity[2]), location=Vector3(z=1000)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

        # NEURAL NET OUTPUTS
        with self.torch.no_grad():
            outputs = self.bot_list[self.brain].forward(*inputs)
        self.controller_state = self.output_formatter.format_model_output(outputs, [packet])[0]

        # KILL
        if (my_car.physics.location.z < 100 or my_car.physics.location.z > 1950 or my_car.physics.location.x < -4000 or
            my_car.physics.location.x > 4000 or my_car.physics.location.y > 5000
                or my_car.physics.location.y < -5000) and self.frame > 50:
            self.frame = self.max_frames

        # LOOPS
        self.frame += 1
        if self.frame >= self.max_frames:
            self.frame = 0

            self.calc_min_fitness()

            self.attempt += 1
            if self.attempt >= 5:
                self.attempt = 0

                self.calc_fitness()

                self.brain += 1  # change bot every reset
                if self.brain >= self.pop:
                    self.brain = 0  # reset bots after all have gone

                    self.next_generation()

                    self.gen += 1

            self.controller_state = SimpleControllerState()  # reset controller
            self.reset()  # reset at start

        return self.controller_state

    def calc_min_fitness(self):
        # CALCULATE MINIMUM DISTANCE TO BALL FOR EACH ATTEMPT
        self.min_distance_to_ball.append(min(self.distance_to_ball))
        self.distance_to_ball = [math.inf] * self.max_frames

    def calc_fitness(self):
        # CALCULATE AVERAGE OF MINIMUM DISTANCE TO BALL FOR EACH GENOME
        sum1 = sum(self.min_distance_to_ball)
        sum1 /= len(self.min_distance_to_ball)

        self.bot_fitness[self.brain] = sum1
        self.min_distance_to_ball = []

    def next_generation(self):
        self.avg_best_fitness()
        self.calc_fittest()

        # PRINT GENERATION INFO
        print("")
        print("     GEN = " + str(self.gen + 1))
        print("-------------------------")
        print("FITTEST = BOT " + str(self.fittest + 1))
        print("------FITNESS = " + str(self.bot_fitness[self.fittest]))
        # print("------WEIGHTS = " + str(self.bot_list[self.fittest]))
        for i in range(len(self.bot_list)):
            print("FITNESS OF BOT " + str(i + 1) + " = " + str(self.bot_fitness[i]))
        print("------MUTATION RATE = " + str(self.mut_rate))

        # NE Functions
        self.selection()
        self.mutate()

    def avg_best_fitness(self):
        # CALCULATE AVG FITNESS OF 5 FITTEST (IDENTICAL) GENOMES
        self.bot_fitness[-self.num_best:] = [sum(self.bot_fitness[-self.num_best:]) / self.num_best] * self.num_best

    def calc_fittest(self):
        temp = math.inf
        for i in range(len(self.bot_list)):
            if self.bot_fitness[i] < temp:
                temp = self.bot_fitness[i]
                self.fittest = i
        return self.fittest

    def reset(self):
        pos = self.ball_set[self.attempt][0]
        # RESET TRAINING ATTRIBUTES AFTER EACH GENOME
        ball_state = BallState(Physics(location=Vector3(pos[0], pos[1], pos[2])))
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=33,
                             physics=Physics(rotation=Rotator(45, 8, 0), velocity=Vector3(0, 0, 0),
                                             angular_velocity=Vector3(0, 0, 0), location=Vector3(0.0, -4608, 500)))
        game_info_state = GameInfoState(game_speed=1)
        game_state = GameState(ball=ball_state, cars={self.index: car_state}, game_info=game_info_state)
        self.set_game_state(game_state)
        time.sleep(0.1)

    def selection(self):
        # COPY FITTEST WEIGHTS TO ALL GENOMES
        state_dict = self.bot_list[self.fittest].state_dict()
        for bot in self.bot_list:
            bot.load_state_dict(state_dict)

    def mutate(self):
        # MUTATE FIRST GENOMES
        self.mut_rate = self.bot_fitness[self.fittest] / 10000
        for i, bot in enumerate(self.bot_list[:-self.num_best]):
            new_genes = self.Model()
            for param, param_new in zip(bot.parameters(), new_genes.parameters()):
                mask = self.torch.rand(param.data.size()) < self.mut_rate / (i + 1)
                param.data[mask] = param_new.data[mask]


def draw_debug(renderer, action_display):
    renderer.begin_rendering()
    renderer.draw_string_2d(10, 10, 4, 4, action_display, color=renderer.white())
    renderer.end_rendering()
