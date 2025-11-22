import torch

from TrainingGround import TrainingGround

torch.set_float32_matmul_precision('high')

def main():
    tg = TrainingGround()
    tg.start()

if __name__ == "__main__":
    main()