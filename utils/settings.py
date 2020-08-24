import os
from os.path import join, abspath, dirname
import platform


PROJ_DIR = join(abspath(dirname(__file__)), '..')
if os.name == 'nt':
    DATA_DIR = 'C:/Users/zfj/research-data/oag-kdd19-data/data'
    OUT_DIR = 'C:/Users/zfj/research-out-data/oag-kdd19-data'
else:
    if platform.system() == "Darwin":
        DATA_DIR = '/Users/zfj/research-data/oag-kdd19-data/data'
        OUT_DIR = '/Users/zfj/research-out-data/oag-kdd19-data'
    else:
        DATA_DIR = '/home/zfj/research-data/oag-kdd19-data/data'
        OUT_DIR = '/home/zfj/research-out-data/oag-kdd19-data'

AUTHOR_DATA_DIR = join(DATA_DIR, 'authors')
PAPER_DATA_DIR = join(DATA_DIR, 'papers')
VENUE_DATA_DIR = join(DATA_DIR, 'venues')
AFF_DATA_DIR = join(DATA_DIR, 'aff')
DOM_ADAPT_DIR = join(DATA_DIR, "dom-adpt")

os.makedirs(OUT_DIR, exist_ok=True)
OUT_PAPER_DIR = join(OUT_DIR, 'paper')
os.makedirs(OUT_PAPER_DIR, exist_ok=True)
OUT_VENUE_DIR = join(OUT_DIR, 'venue')
os.makedirs(OUT_VENUE_DIR, exist_ok=True)


AUTHOR_TYPE = 0
PAPER_TYPE = 1
VENUE_TYPE = 2

MAX_WORD_TOKEN_NUM = 99999
