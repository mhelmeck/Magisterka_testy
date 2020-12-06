from datetime import datetime

from consts import MODEL_SAVE_PATH


def get_save_model_path():
    return MODEL_SAVE_PATH.format('unet_128', datetime.now().strftime("%d%m%Y_%H%M"))