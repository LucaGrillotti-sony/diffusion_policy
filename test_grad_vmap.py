import copy

import dill
import hydra
import torch

from diffusion_policy.policy.diffusion_guided_ddim import DDIMGuidedScheduler
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def func(array_actions):
    differences = array_actions[1:] - array_actions[:-1]
    distances = torch.norm(differences, dim=-1)
    mean_distance = torch.mean(distances)
    return mean_distance


def test():
    array_actions = torch.tensor([
        [[1., 1.],
        [1., 2.],
        [5., 5.],
        [5., 5.],],
        [[1., 1.],
         [1., 2.],
         [5., 5.],
         [5., 5.], ],
        [[1., 1.],
         [1., 2.],
         [5., 5.],
         [5., 5.], ]
    ], requires_grad=True)
    output = torch.vmap(func)(array_actions)
    print(len(output.shape), len(array_actions.shape))

    output = output.mean()
    print("output.requires_grad", output.requires_grad)
    grad = torch.autograd.grad(output,array_actions)[0]
    array_actions.requires_grad = False
    print(array_actions.requires_grad)

    print(grad)

def main(args=None):
    ckpt_path = "/home/lucagrillotti/ros/humble/src/diffusion_policy/data/outputs/2024.01.16/19.33.40_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    _config_noise_scheduler = {**copy.deepcopy(cfg.policy.noise_scheduler)}
    del _config_noise_scheduler["_target_"]
    workspace.model.noise_scheduler = DDIMGuidedScheduler(**_config_noise_scheduler)

if __name__ == '__main__':
    test()
    # main()