"""Constants for interprocess communication

Author : tommoral <thomas.moreau@inria.fr>
"""

# Inter-process communication constants
TAG_ROOT = 4242

# Worker control flow
TAG_WORKER_STOP = 0
TAG_WORKER_RUN_DICOD = 1
TAG_WORKER_RUN_DICODILE = 2

# DICOD worker control messages
TAG_DICOD_STOP = 8
TAG_DICOD_UPDATE_BETA = 9
TAG_DICOD_PAUSED_WORKER = 10
TAG_DICOD_RUNNING_WORKER = 11
TAG_DICOD_INIT_DONE = 12

# DICODILE worker control tags
TAG_DICODILE_STOP = 16
TAG_DICODILE_COMPUTE_Z_HAT = 17
TAG_DICODILE_GET_COST = 18
TAG_DICODILE_GET_Z_HAT = 19
TAG_DICODILE_GET_Z_NNZ = 20
TAG_DICODILE_GET_SUFFICIENT_STAT = 21
TAG_DICODILE_SET_D = 22
TAG_DICODILE_SET_SIGNAL = 23
TAG_DICODILE_SET_PARAMS = 24
TAG_DICODILE_SET_TASK = 25


# inter-process message size
SIZE_MSG = 4


# Output control
GLOBAL_OUTPUT_TAG = "\r[{level_name}:DICOD-{identity}] "
WORKER_OUTPUT_TAG = "\r[{level_name}:DICOD:Worker-{identity:<3}] "


# Worker status
STATUS_STOP = 0
STATUS_PAUSED = 1
STATUS_RUNNING = 2
STATUS_FINISHED = 4
