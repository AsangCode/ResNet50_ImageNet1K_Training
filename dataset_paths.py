import os

DATA_DIR = os.getenv('DATASET_PATH', '/mnt/imagenet/extracted')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
