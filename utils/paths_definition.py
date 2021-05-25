from utils.os_variable_utils import get_profile_name

CASE_IMAGES_DIR = {
    'local-a': '../kits19/extracted_data/destination_images/',
    'local-m': '../kits19/extracted_data/destination_images/',
    'cloud': '/home/jupyter/data/case_images/'
}

CASE_MASKS_DIR = {
    'local-a': '../kits19/extracted_data/destination_masks/',
    'local-m': '../kits19/extracted_data/destination_masks/',
    'cloud': '/home/jupyter/data/case_masks/'
}
CASE_IMAGES_WITH_MASKS_DIR = {
    'local-a': '../kits19/extracted_data/destination_images_with_masks/',
    'local-m': '../kits19/extracted_data/destination_images_with_masks/',
    'cloud': ''
}
MODEL_SAVE_DIR = {
    'local-a': 'training_result/training_{}_{}/',
    'local-m': 'training_result/training_{}_{}/',
    'cloud': '/home/jupyter/training_results/training_{}_{}/'
}
LOGS_DIR = {
    'local-a': 'logs/',
    'local-m': 'logs/',
    'cloud': '/home/jupyter/logs/'
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
