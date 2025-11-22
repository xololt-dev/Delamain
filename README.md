# Delamain
Repository dedicated towards models trained and evaluated on OpenAI Gymnasium Car Racing enviroment.

## ROCm
Repository includes helper script(s) for running code inside [official ROCm docker container](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html).

```
bash ./rocm-pytorch/delamain_setup.sh [docker_container_id] [destination_folder_for_backup]
```

> [!TIP]
> If your GPU isn't officially supported, you can try overriding the enviroment variable HSA_OVERRIDE_GFX_VERSION before calling main.py script ([GFX versions](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html#architecture-support-compatibility-matrix)).

> [!IMPORTANT]
> Call delamain_setup.sh script from project root folder.

## References
### Inspired by:
[wiitt](https://github.com/wiitt) [implementation](https://github.com/wiitt/DQN-Car-Racing)

### Used in the project:
[PyTorch](https://docs.pytorch.org/docs/stable/index.html)

[numpy](https://numpy.org/doc/stable/)

[Gymnasium - Car Racing enviroment](https://gymnasium.farama.org/environments/box2d/car_racing/)

[pyyaml](https://pyyaml.org/)