
# Baynex
sudo docker run -d -it \
    -v ${PWD}:/workspace/mlsys \
    -v /docker/data/cheezestick:/data \
    --name sk-se3 \
    --net=host \
    --ipc=host \
    --gpus all \
    nvcr.io/nvidia/pytorch:22.10-py3 bash