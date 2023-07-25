CULANEROOT="$HOME/datasets/CULane
cd $CULANEROOT
for file in `$(ls *.tar.gz)`
do
    echo "found '$file' in '$CULANEROOT'"
    tar xf $file
done