docker run -it \
          --cap-add=SYS_PTRACE \
          --security-opt seccomp=unconfined \
          --device=/dev/kfd \
          --device=/dev/dri \
          --group-add video \
          --ipc=host \
          --shm-size 12G \
          --cpus=6 \
          rocm/pytorch:latest
