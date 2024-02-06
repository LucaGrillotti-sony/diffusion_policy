"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import pathlib
import sys

import dill
import torch

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy', 'config'))
)
def main(new_cfg):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    # OmegaConf.resolve(cfg)

    # workspace: TrainDiffusionUnetHybridWorkspace = cls(cfg)
    print(1)

    ckpt_path = "/home/lucagrillotti/ros/humble/src/diffusion_policy/data/outputs/2024.01.31/19.22.29_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"  # trained to also optimize actions
    print(2)

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: TrainDiffusionUnetHybridWorkspace = cls(new_cfg)
    print(3)

    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    print(4)
    old_model = workspace.model
    old_optimizer = workspace.optimizer
    print("OLD MODEL", old_model.eta_coeff_critic)

    OmegaConf.resolve(new_cfg)

    cls = hydra.utils.get_class(new_cfg._target_)
    new_workspace: BaseWorkspace = cls(new_cfg)

    #  /!\ Warning: also copy optimizer!!
    new_workspace.model = old_model
    new_workspace.optimizer = old_optimizer

    print("NEW MODEL", new_workspace.model.eta_coeff_critic)
    new_workspace.run()


if __name__ == "__main__":
    print(0)
    main()
