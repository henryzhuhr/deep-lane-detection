
proj_home=$(pwd)

DEFAULT=$(echo -en '\033[0m')
RED=$(echo -en '\033[00;31m')
GREEN=$(echo -en '\033[00;32m')
YELLOW=$(echo -en '\033[00;33m')
BLUE=$(echo -en '\033[00;34m')
MAGENTA=$(echo -en '\033[00;35m')
PURPLE=$(echo -en '\033[00;35m')
CYAN=$(echo -en '\033[00;36m')
LIGHTGRAY=$(echo -en '\033[00;37m')
LRED=$(echo -en '\033[01;31m')
LGREEN=$(echo -en '\033[01;32m')
LYELLOW=$(echo -en '\033[01;33m')
LBLUE=$(echo -en '\033[01;34m')
LMAGENTA=$(echo -en '\033[01;35m')
LPURPLE=$(echo -en '\033[01;35m')
LCYAN=$(echo -en '\033[01;36m')
WHITE=$(echo -en '\033[01;37m')

if [ "$(uname)" = "Darwin" ]; then
    NUM_CORES=$(sysctl -n hw.ncpu)
elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
    NUM_CORES=$(nproc --all)
else
    NUM_CORES=4
fi

mkdir -p $proj_home/libs
# Nano 上可以 sudo apt-get install libgflags-dev
if [ ! -d "$proj_home/libs/gflags" ]; then
    mkdir -p $proj_home/temps
    cd $proj_home/temps
    git clone -b v2.2.2 --depth=1 https://github.com/gflags/gflags.git gflags
    cd $proj_home/temps/gflags
    mkdir build_
    cd build_
    cmake .. -DCMAKE_INSTALL_PREFIX=$proj_home/libs/gflags
    make -j${NUM_CORES}
    make install
fi
if [ -d "$proj_home/libs/gflags/lib/cmake/gflags" ]; then
    echo -e "${GREEN}gflags build success${DEFAULT}"
    export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$proj_home/libs/gflags/lib/cmake/gflags
else
    echo -e "${RED}gflags build failed${DEFAULT}"
    exit 1
fi

cp -r ../weights $proj_home
    

mkdir $proj_home/build
cd $proj_home/build
rm CMakeCache.txt
cmake ..
make -j${NUM_CORES}
make install

echo ""
echo "${LCYAN} RUN: ${DEFAULT}"
echo ""
cd $proj_home/bin
./infer-onnx-image