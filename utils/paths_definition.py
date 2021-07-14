from utils.consts import CASE_PATTERN, PNG_PATTERN
from utils.os_variable_utils import get_profile_name

CASE_IMAGES_DIR = {
    'local-a': 'D:/ALEKSY/data_mgr/extracted_data/destination_images/',
    'local-m': '../kits19/extracted_data/destination_images/',
    'cloud': '/home/maciek/data/case_images/'
}

CASE_MASKS_DIR = {
    'local-a': 'D:/ALEKSY/data_mgr/extracted_data/destination_masks/',
    'local-m': '../kits19/extracted_data/destination_masks/',
    'cloud': '/home/maciek/data/case_masks/'
}
CASE_IMAGES_WITH_MASKS_DIR = {
    'local-a': '/mnt/d/ALEKSY/data_mgr/extracted_data/destination_images_with_masks/',
    'local-m': '../kits19/extracted_data/destination_images_with_masks/',
    'cloud': ''
}
MODEL_SAVE_DIR = {
    'local-a': 'training_result/training_{}_{}/',
    'local-m': 'training_result/training_{}_{}/',
    'cloud': '/home/maciek/training_results/training_{}_{}/'
}
LOGS_DIR = {
    'local-a': 'logs/',
    'local-m': 'logs/',
    'cloud': '/home/maciek/logs/'
}
PREDICTION_RESULTS_WITH_MASKS = {
    'local-a': 'D:/ALEKSY/data_mgr/prediction_results/results_for_{}_{}_{}/',
    'local-m': 'prediction_results/results_for_{}_{}_{}/',
    'cloud': '/home/maciek/prediction_results/results_for_{}_{}_{}/',
}


def get_model_save_path():
    return MODEL_SAVE_DIR[get_profile_name()] + '/cp.ckpt'


def get_kidney_masks_dir():
    return CASE_MASKS_DIR[get_profile_name()] + 'kidneys/'


def get_tumor_masks_dir():
    return CASE_MASKS_DIR[get_profile_name()] + 'tumors/'


def get_images_dir():
    return CASE_IMAGES_DIR[get_profile_name()]


def get_logs_dir():
    return LOGS_DIR[get_profile_name()]


def get_prediction_results_case_dir(model_name, threshold, img_size, case_index):
    return PREDICTION_RESULTS_WITH_MASKS[get_profile_name()].format(model_name, threshold, img_size) + CASE_PATTERN.format(case_index)


def get_prediction_results_image_path(model_name, threshold, img_size, case_index, img_index):
    return get_prediction_results_case_dir(model_name, threshold, img_size, case_index) + '/' + PNG_PATTERN.format(img_index)
