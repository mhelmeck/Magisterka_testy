from datetime import datetime
import os

from utils.os_variable_utils import get_profile_name, get_model_name
from utils.paths_definition import get_model_save_path


def get_save_model_path(model_name):
    return get_model_save_path().format(model_name, datetime.now().strftime("%d%m%Y_%H%M"))


def load_variables():
    channel_numbers = 3
    profile = get_profile_name()
    model_name = get_model_name()
    img_size = int(os.environ.get('IMG_SIZE')) if os.environ.get('IMG_SIZE') else 512
    epochs = int(os.environ.get('EPOCHS')) if os.environ.get('EPOCHS') else 512
    batch_size = int(os.environ.get('BATCH_SIZE')) if os.environ.get('BATCH_SIZE') else 32
    starts_neuron = int(os.environ.get('STARTS_NEURON')) if os.environ.get('STARTS_NEURON') else 16

    start_case_index_train = int(os.environ.get('START_CASE_INDEX_TRAIN')) if os.environ.get('START_CASE_INDEX_TRAIN') else 0
    end_case_index_train = int(os.environ.get('END_CASE_INDEX_TRAIN')) if os.environ.get('END_CASE_INDEX_TRAIN') else 180

    start_case_index_test = int(os.environ.get('START_CASE_INDEX_TEST')) if os.environ.get('START_CASE_INDEX_TEST') else 181
    end_case_index_test = int(os.environ.get('END_CASE_INDEX_TRAIN')) if os.environ.get('END_CASE_INDEX_TRAIN') else 209

    print(
        'Executing model with parameters: \n'
        'PROFILE = %s\n' % profile,
        'MODEL_NAME = %s\n' % model_name,
        'IMG_SIZE = %s\n' % img_size,
        'EPOCHS = %s\n' % epochs,
        'BATCH_SIZE = %s\n' % batch_size,
        'STARTS_NEURON = %s\n' % starts_neuron,
        'START_CASE_INDEX_TRAIN = %s\n' % start_case_index_train,
        'END_CASE_INDEX_TRAIN = %s\n' % end_case_index_train,
        'START_CASE_INDEX_TEST = %s\n' % start_case_index_test,
        'END_CASE_INDEX_TRAIN = %s\n' % end_case_index_test
    )

    return (model_name, channel_numbers, img_size, epochs, batch_size, starts_neuron, start_case_index_train,
            end_case_index_train,
            start_case_index_test, end_case_index_test)