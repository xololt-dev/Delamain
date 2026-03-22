FROM rocm/pytorch:latest

WORKDIR /Delamain

# Set Environment Variables for ROCm
# This helps PyTorch find the GPU library paths automatically
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:$LD_LIBRARY_PATH
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

RUN apt-get update && apt-get install -y swig \
    && rm -rf /var/lib/apt/lists/*

COPY delamain_req.txt /Delamain/
RUN pip3 install -r /Delamain/delamain_req.txt

COPY alternative_models /Delamain/alternative_models
COPY enviroment /Delamain/enviroment
COPY tests /Delamain/tests
COPY main.py /Delamain/
COPY training_params.yaml /Delamain/
COPY TrainingGround.py /Delamain/

RUN mkdir -p /Delamain/training/logs

CMD ["bash"]