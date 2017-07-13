import os

IMAGE_CHANNEL = 3
IMAGE_SIZE = 250
NUM_CLASSES = 10
NUM_EXAMPLES = 10
BATCH_SIZE = 3
FORMAT = 'jpg'

PATH_IMAGES = 'data_source'
PATH_LABELS = os.path.join(PATH_IMAGES, 'labels.txt')
PATH_RECORD = os.path.join('data_target', 'MyRecord')