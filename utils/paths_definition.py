from utils.os_variable_utils import get_profile_name

CASE_IMAGES_DIR = {
    'local': '../destination_clean/',
    'cloud': '/home/jupyter/data/case-images/'
}

CASE_MASKS_DIR = {
    'local': '../destination_mask/',
    'cloud': '/home/jupyter/data/case-masks/'
}
CASE_IMAGES_WITH_MASKS_DIR = {
    'local': '../destination_with_label/',
    'cloud': ''
}
MODEL_SAVE_DIR = {
    'local': 'training_result/training_{}_{}/',
    'cloud': '/home/jupyter/training_results/'
}
LOGS_DIR = {
    'local': 'logs/',
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
