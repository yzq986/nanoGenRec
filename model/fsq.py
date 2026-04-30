# shim: module moved to tokenizer/fsq.py
from tokenizer.fsq import *  # noqa: F401, F403
from tokenizer.fsq import (FSQLayer, LearnedFSQLayer, fsq_layer_from_state,
                            FSQ_LEVEL_CONFIGS, _codebook_size)
