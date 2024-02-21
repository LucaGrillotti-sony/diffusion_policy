"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import dataclasses
import logging
import os
import shlex
import subprocess
import sys
import time
from typing import Optional, List, Tuple, Protocol

from hydra import initialize, compose

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import dartlib
import structlog
from dart_client.metrics import get_metric_writer
from dartlib.dart2.run_config.private import get_private_run_params
from saiconf import make_config_registry_root, register_config


import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

logger = logging.getLogger(__name__)
LOG = structlog.getLogger()


@make_config_registry_root
class PreOp(Protocol):
    """
    Base PreOp registry object
    """

    def execute(self) -> None:
        pass


@register_config("bash")
@dataclasses.dataclass(frozen=True)
class BashPreOp(PreOp):
    """
    A preop that lets you specify a bash file to be executed.

    Note: The original code for this PreOp allowed direct execution of code on the shell.
    This update requires that the code you want executed must be in a script in the git repo
    (some child path of the working directory). The means that whoever launches the run must
    therefore be someone who has git authorization.
    """

    bash_script: pathlib.Path
    script_args: str = ""

    def execute(self) -> None:
        LOG.info(f"Executing BashPreOp: \"{self.bash_script}\"")
        start_time = time.time()
        args = shlex.split(self.script_args)
        subprocess.check_call(["bash", str(self.bash_script), *args], shell=False)
        LOG.info(f"BashPreOp complete ({time.time() - start_time}s)")

    def __post_init__(self) -> None:
        assert self.bash_script.suffix in [".sh", ".bash"], f"Invalid {self.bash_script=}"
        # assert self.bash_script.resolve().relative_to(
        #     pathlib.Path.cwd()
        # ), f"Bash script is not on a subpath of cwd ({pathlib.Path.cwd()})"


@register_config("s3cp")
@dataclasses.dataclass(frozen=True)
class S3cp(PreOp):
    """
    Pull down data from and s3 bucket into local storage using `aws s3 cp`
    """

    # Ex: sai-shokunin
    s3_bucket: str

    """
    Specify directories to download recursively.

    Format: s3_target:local_dest

    Ex:

    saic_depth_completion:dataset

    If we have s3 objects:
    saic_depth_completion/dir1/file1.txt
    saic_depth_completion/dir1/subdir/subfile1.txt

    Then locally we will have
    ./dataset/dir1/file1.txt
    ./dataset/dir1/subdir/subfile1.txt

    """
    s3_mounts: List[str] = dataclasses.field(default_factory=list)

    """
    Specify individual files to download.
    Format: s3_target:local_dest
    """
    s3_files: List[str] = dataclasses.field(default_factory=list)

    def execute(self) -> None:
        """
        Check all the registered mounts and pull them down from s3 sequentially.
        @return:
        """

        def format_src_dest(original: str) -> Tuple[str, str]:
            splits = original.split(":")
            assert len(splits) == 2, f"Invalid s3 mount ({original}). Format s3_mount:local_target_dir"
            return f"s3://{self.s3_bucket}/{splits[0]}", splits[1]

        LOG.info("Running S3cp startup")
        start_time = time.monotonic()
        for mount in self.s3_mounts:
            src, target = format_src_dest(mount)
            # we will want exceptions to bubble up
            subprocess.check_call(['aws', 's3', 'cp', src, target, '--recursive'])

        for filename in self.s3_files:
            src, target = format_src_dest(filename)
            subprocess.check_call(['aws', 's3', 'cp', src, target])

        LOG.info(f"S3cp startup complete ({time.monotonic() - start_time}s)")


@dataclasses.dataclass(frozen=True)
class Config:

    """
    The python script that is being wrapped and corresponding command line arguments to call.
    """

    # python_script: pathlib.Path
    # python_args: str = ""

    # optionally set the working directory before any other operations.
    working_dir: Optional[pathlib.Path] = None

    """
    local directories to watch for artifacts that will be copied to the s3 bucket associated with this run

    Format can be either:
    1. source_dir  : In this case the files will written directly to the s3 target associated with the run.
    Ex. source_dir/temp.txt -> s3_target/temp.txt

    2. source_dir:s3_folder  : In this case we append s3_folder to s3_target.
    Ex. source_dir/temp.txt -> s3_target/s3_folder/temp.txt
    """
    artifact_watch_dirs: Optional[List[str]] = None

    # a list of operations to run first. Use this for any setup that needs to happen. Ex. downloading a dataset
    preops: Optional[List[PreOp]] = None

    # If specified this directory will be watched (recursively) for tensorboard files,
    # metrics will be read and written to dart metrics.
    tensorboard_dir: Optional[str] = None

    def __post_init__(self) -> None:

        if self.artifact_watch_dirs:
            for artifact_watch_dir in self.artifact_watch_dirs:
                assert len(artifact_watch_dir.split(":")) <= 2, f"Invalid artifact_watch_dir: {artifact_watch_dir}"

        # assert self.python_script.suffix == ".py", f"Invalid file type {self.python_script.suffix}"
        # assert self.python_script.resolve().relative_to(
        #     pathlib.Path.cwd()
        # ), f"python_script not on subpath of cwd ({pathlib.Path.cwd()})"


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


def main_wrapper():
    # unpack the run_config which will contain various settings for the dart run itself
    run_config = dartlib.dart2.run_config.init_run_config()
    params = get_private_run_params(run_config)

    # The app config is the information contained in 'script_config' in the config file.
    # See Config for valid params
    app_config: Config = dartlib.dart2.util.read_app_config(Config)

    if app_config.working_dir:
        os.chdir(app_config.working_dir)

    # Run any pre ops that are specified for setting up the env.
    if app_config.preops:
        for pre_op in app_config.preops:
            pre_op.execute()

    # then run main after removing current arguments from the stack
    sys.argv = [sys.argv[0]]
    main()

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        logger.info('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logger.info('{}{}'.format(subindent, f))


if __name__ == "__main__":
    main_wrapper()
