import os

# DATA
dataset = 'CULane'
data_root = os.getenv('CULANEROOT')

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = False
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = "tmps"

# FINETUNE or RESUME MODEL PATH
finetune = "weights/culane_18.pth"
resume = None

# TEST
test_model = "weights/culane_18.pth"
test_work_dir = "tmps"

num_lanes = 4
