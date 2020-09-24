import os
from os.path import join as oj

DIR_FILE = os.path.dirname(os.path.realpath(__file__))
DIR_REPO = oj(DIR_FILE, '..') # directory of the config file
DIR_FIGS = oj(DIR_REPO, 'reports', 'figs')
DIR_IMAGENET_DATA = '/scratch/users/vision/data/cv/imagenet_full'