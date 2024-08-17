export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
export OMP_NUM_THREADS=1 # you can change this value according to your number of cpu cores

python3 -m torch.distributed.launch --nproc_per_node=$NGPUS train.py temp/culane.py

