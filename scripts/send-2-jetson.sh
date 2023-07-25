export JETSON_NANO_USER=nano
export JETSON_NANO_HOST=192.168.101.225
# 提前在Jetson Nano上创建好目录
TARGET_DIR=project/lane-detection

echo "JETSON_NANO_USER: $JETSON_NANO_USER"
echo "JETSON_NANO_HOST: $JETSON_NANO_HOST"
echo "TARGET_DIR: $TARGET_DIR"



rm -rf deploy/utils/__pycache__
find . -name ".DS_Store" -depth -exec rm {} \;


# scp -r weights/* $JETSON_NANO_USER@$JETSON_NANO_HOST:~/$TARGET_DIR
# scp -r deploy/* $JETSON_NANO_USER@$JETSON_NANO_HOST:~/$TARGET_DIR

# scp $HOME/Downloads/IMG_5281.MOV $JETSON_NANO_USER@$JETSON_NANO_HOST:~/$TARGET_DIR

# scp -r scripts/onn2trt.sh $JETSON_NANO_USER@$JETSON_NANO_HOST:~/$TARGET_DIR



scp -r deploy/infer-trtEngine.py $JETSON_NANO_USER@$JETSON_NANO_HOST:~/$TARGET_DIR
scp -r deploy/utils $JETSON_NANO_USER@$JETSON_NANO_HOST:~/$TARGET_DIR
scp -r temp $JETSON_NANO_USER@$JETSON_NANO_HOST:~/$TARGET_DIR


# 将生成的engine文件拷贝回来
# scp $JETSON_NANO_USER@$JETSON_NANO_HOST:~/$TARGET_DIR/culane_18.engine  weights/
