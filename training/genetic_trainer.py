from rlbottraining.exercise_runner import run_playlist
from rlbottraining.training_exercise import TrainingExercise

from rlbot.matchcomms.common_uses.set_attributes_message import make_set_attributes_message
from rlbot.matchcomms.common_uses.reply import send_and_wait_for_replies
from rlbot.training.training import Grade, Pass
from rlbot.utils.logging_utils import get_logger
from rlbot.setup_manager import setup_manager_context

from typing import Optional, Callable

from torch.nn import Module

import math
import io
from multiprocessing.reduction import ForkingPickler
import pickle
import torch
import os
import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)  # this is for first process imports

from framework.self_evolving_car.genetic_algorithm import GeneticAlgorithm
from training.linkuru_playlist import make_default_playlist
from examples.levi.torch_model import SymmetricModel


def create_on_briefing(send_model: Module) -> Callable:
    def on_briefing(self: TrainingExercise) -> Optional[Grade]:
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(send_model)
        _ = send_and_wait_for_replies(self.get_matchcomms(), [
            make_set_attributes_message(0, {'model_hex': buf.getvalue().hex()}),
        ])
        return None
    return on_briefing

if __name__ == '__main__':
    logger = get_logger('genetic algorithm')
    ga = GeneticAlgorithm()
    gen = 0 # Init generation count
    bot_count = 10
    fitness = [0]*bot_count # Init fitness matrix
    fittest = math.inf
    models = [SymmetricModel().share_memory() for _ in range(bot_count)]
    mut_rate = 0.2
    num_fittest = 5

    with setup_manager_context() as setup_manager:
        while True:
            gen += 1 # Generation count up 1
            bot = 0  # Init bot count

            while bot < bot_count:

                playlist = make_default_playlist(create_on_briefing(models[bot]))[0:1] # Create scenario

                result = next(run_playlist(playlist, setup_manager=setup_manager)) # Get result
                fitness[bot] = result.grade.loss # Assign Fitness
                logger.info("BOT: "+str(bot)+" "+str(fitness[bot]))#+str(models[bot].state_dict())) # Log Info
                fittest = ga.calc_fittest(fitness) # Calculate fittest bot

                bot += 1

                if isinstance(result.grade, Pass):
                    torch.save(model.state_dict(), f'exercise_{result.reproduction_info.playlist_index}.mdl')
                    break

            ga.crossover(models[fittest], models)
            ga.mutate(models[:num_fittest], mut_rate)

            print("GENERATION: "+str(gen)+" | "+"FITTEST: "+str(fittest))