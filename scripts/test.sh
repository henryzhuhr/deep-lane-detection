
log_dir=tmps/20230712_145732_lr_1e-02_b_4
python3 test.py \
    $log_dir/culane-finetune.py \
    --test_model $log_dir/ufld-final.pt

