DEST_IMAGES_PATH = 'destination_clean/'
DEST_IMAGES_WITH_MASKS_PATH = 'destination_with_label/'
DEST_MASKS_PATH = 'destination_mask/'
DEST_KIDNEY_MASKS_PATH = DEST_MASKS_PATH + 'kidneys/'
DEST_TUMOR_MASKS_PATH = DEST_MASKS_PATH + 'tumors/'
CASE_PATTERN = 'case_{:05d}'
PNG_PATTERN = '{:05d}.png'
PNG_IN_CASE_PATTERN = CASE_PATTERN + '/' + PNG_PATTERN
MODEL_SAVE_DIR = 'training_{}_{}/'
LOGS_DIR = 'logs/'
