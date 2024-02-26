import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace


@hydra.main(
    version_base=None,
    config_path='.',
    config_name='franka_end_effector_image_obs',
)
def main(cfg: OmegaConf):

    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)

    import os

    # get the current working directory
    current_working_directory = os.getcwd()

    # print output to the console
    print("Working dir", current_working_directory)

    workspace: BaseWorkspace = cls(cfg)

    workspace.run()

if __name__ == '__main__':
    main()