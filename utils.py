from datetime import datetime

from consts import MODEL_SAVE_PATH


def get_save_model_path(model_name):
    return MODEL_SAVE_PATH.format(model_name, datetime.now().strftime("%d%m%Y_%H%M"))
