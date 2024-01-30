"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

import dill
import torch

from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    # config_path=str(pathlib.Path(__file__).parent.joinpath(
    #     'diffusion_policy','config'))
)
def main():
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    # OmegaConf.resolve(cfg)

    # workspace: TrainDiffusionUnetHybridWorkspace = cls(cfg)

    ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.01.26/15.37.47_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"  # trained to also optimize actions
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: TrainDiffusionUnetHybridWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace.run_train_classifier()


if __name__ == "__main__":
    main()
