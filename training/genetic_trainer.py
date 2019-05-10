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

def mutate(list, mut_rate):
    """Randomizes a certain amount of the first five models' parameters based on mutation rate
    :param list contains the parameters to be mutated
    :param mut_rate is the mutation rate"""

    for i, bot in enumerate(list):
        new_genes = SymmetricModel()
        for param, param_new in zip(bot.parameters(), new_genes.parameters()):
            mask = torch.rand(param.data.size()) < mut_rate / (i + 1)
            param.data[mask] = param_new.data[mask]

def calc_fittest(list):
    """calculates the fittest bot of each generation
    :param list is input array to find minimum
    :returns tuple of the fittest score and index """

    index = 0
    temp = math.inf
    for count, i in enumerate(list):
        if i < temp:
            temp = i
            index = count
    return index

if __name__ == '__main__':
    logger = get_logger('genetic algorithm')

    gen = 0 # Init generation count
    bot_count = 10
    fitness = [0]*bot_count # Init fitness matrix
    fittert = 0
    models = [SymmetricModel()] * bot_count
    mut_rate = 0.2

    # model.load_state_dict(torch.load(f'exercise_0.mdl'))
    for model in models:
        model.share_memory()
        model.load_state_dict(SymmetricModel().state_dict())  # Load random model

    with setup_manager_context() as setup_manager:
        while True:
            gen += 1 # Generation count up 1
            bot = 0  # Init bot count

            while bot < bot_count:

                playlist = make_default_playlist(create_on_briefing(models[bot]))[0:1] # Create scenario

                result = next(run_playlist(playlist, setup_manager=setup_manager)) # Get result
                fitness[bot] = result.grade.loss # Assign Fitness
                logger.info("BOT: "+str(bot)+" "+str(fitness[bot])) # Log Info
                fittest = calc_fittest(fitness) # Calculate fittest bot

                bot += 1

            for model in models:
                model.state_dict = models[fittest].state_dict

            mutate(models, mut_rate)

            print("GENERATION: "+str(gen)+" | "+"FITTEST: "+str(fittest))