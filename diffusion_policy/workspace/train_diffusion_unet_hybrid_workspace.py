import math

from diffusion_policy.model.diffusion.conditional_unet1d_critic import DoubleCritic
from diffusion_policy.networks.classifier import ClassifierStageScooping
from diffusion_policy.networks.trainer_classifier import TrainerClassifier

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to, custom_tree_map
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.lagrange_parameter = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)

        self.lagrange_optimizer = hydra.utils.instantiate(
            cfg.lagrange_optimizer, params=[self.lagrange_parameter])

        self.eps_lagrange_constraint_mse_predictions = cfg.eps_lagrange_constraint_mse_predictions

        # Critic networks and optimizer: todo
        self.critic: DoubleCritic = hydra.utils.instantiate(cfg.critic, obs_feature_dim=self.model.obs_feature_dim)
        self.critic_optimizer = hydra.utils.instantiate(
            cfg.critic_optimizer, params=self.critic.parameters()
        )
        self.critic_target = copy.deepcopy(self.critic)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        self.critic.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        self.critic_target.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        critic_target = hydra.utils.instantiate(
            cfg.critic_target,
            model=self.critic_target)


        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        dict_logging = {**cfg.logging}
        dict_logging["name"] = str(self.output_dir)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **dict_logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.lagrange_parameter.to(device)
        self.critic.to(device)

        if self.ema_model is not None:
            self.ema_model.to(device)
        self.critic_target.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        sigmoid_lagrange = self.get_sigmoid_lagrange(detach=True).to(cfg.training.device)
                        raw_loss_actor, _metrics_training, _other_data_model = self.model.compute_loss(batch, sigmoid_lagrange)
                        # raw_critic_loss, _metrics_critic_training = self.model.compute_critic_loss(batch, ema_model=self.ema_model)

                        # loss = (raw_loss + raw_critic_loss) / cfg.training.gradient_accumulate_every
                        loss_actor = raw_loss_actor / cfg.training.gradient_accumulate_every
                        loss_actor.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        raw_loss_lagrange, _metrics_lagrange = self.compute_loss_lagrange(sample_actions=_other_data_model["sample_actions"], batch=batch)
                        self.update_lagrange(raw_loss_lagrange)

                        # Update critic
                        loss_critic, metrics_critic = self.critic.compute_critic_loss(batch,
                                                                                      nobs_features=_other_data_model['nobs_features'],
                                                                                      critic_target=self.critic_target,
                                                                                      policy=self.model)
                        loss_critic.backward()
                        self.critic_optimizer.step()
                        self.critic_optimizer.zero_grad()

                        # update ema and critic target
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        critic_target.step(self.critic)

                        # logging
                        raw_loss_cpu = raw_loss_actor.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0],
                            **_metrics_training,
                            **_metrics_lagrange,
                            **metrics_critic,
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        list_metrics = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_actor, _metrics, _ = self.model.compute_loss(batch, sigmoid_lagrange=self.get_sigmoid_lagrange(detach=True))
                                val_losses.append(loss_actor)
                                list_metrics.append(_metrics)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                            dict_metrics = custom_tree_map(
                                lambda *x: torch.mean(torch.tensor(x)).item(),
                                *list_metrics,
                            )
                            step_log.update(dict_metrics)


                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def run_train_classifier(self, num_steps: int = 100):
        NUM_CLASSES = 3
        IN_FEATURES_CLASSIFIER = 135

        cfg = copy.deepcopy(self.cfg)

        device = torch.device(cfg.training.device)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        classifier = ClassifierStageScooping(in_features=IN_FEATURES_CLASSIFIER, number_of_classes=NUM_CLASSES)
        classifier = classifier.to(device)
        self.model = self.model.to(device)

        optimizer_to(self.optimizer, device)

        log_path = os.path.join(self.output_dir, 'logs_classifier.json.txt')

        optimizer = torch.optim.AdamW(classifier.parameters(), betas=(0.95, 0.999), eps=1.0e-08, lr=0.0001, weight_decay=1.0e-06)

        dict_logging = {**cfg.logging, "name": str(self.output_dir), "project": "Classifier Training"}

        checkpoint_every = 10

        wandb_run_classifier = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **dict_logging
        )
        train_sampling_batch = None

        print("device:", device)

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(num_steps):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Classifier Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        labels = batch['label']
                        shape_labels = labels.shape
                        obs_features = self.model.compute_obs_encoding(batch, detach=True)
                        obs_features = obs_features.view(*shape_labels, -1)


                        raw_loss = classifier.compute_loss(obs_features, labels)
                        # accuracy = classifier.accuracy(obs_features, labels)
                        raw_loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run_classifier.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        accuracies = list()
                        # list_metrics = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                       leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                if train_sampling_batch is None:
                                    train_sampling_batch = batch
                                labels = batch['label']
                                shape_labels = labels.shape
                                obs_features = self.model.compute_obs_encoding(batch, detach=True)
                                obs_features = obs_features.view(*shape_labels, -1)

                                val_loss = classifier.compute_loss(obs_features, labels)
                                accuracy = classifier.accuracy(obs_features, labels)
                                val_losses.append(val_loss)
                                accuracies.append(accuracy)
                                # list_metrics.append(_metrics)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            accuracy_mean = torch.mean(torch.tensor(accuracies)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            step_log['accuracy'] = accuracy_mean

                            # dict_metrics = custom_tree_map(
                            #     lambda *x: torch.mean(torch.tensor(x)).item(),
                            #     *list_metrics,
                            # )
                            # step_log.update(dict_metrics)

                # checkpoint
                if (self.epoch % checkpoint_every) == 0:
                    # checkpointing
                    # TODO
                    torch.save(classifier.state_dict(), os.path.join(self.output_dir, 'classifier.pth'))

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value


                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run_classifier.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def compute_loss_lagrange(self, sample_actions, batch, ):
        mse = F.mse_loss(sample_actions['action_pred'], batch['action']).detach().item()  # todo verify if action_predictions are of this form
        lagrange_sigmoid = self.get_sigmoid_lagrange()
        constraint_diff = mse - self.eps_lagrange_constraint_mse_predictions
        constraint_loss = lagrange_sigmoid * constraint_diff / math.fabs(constraint_diff)

        lagrangian = self.lagrange_parameter.data
        sigmoid_lagrange = self.get_sigmoid_lagrange(detach=True)

        metrics = {
            "sigmoid_lagrange": sigmoid_lagrange.item(),
            "lagrange": lagrangian.detach().item(),
            "loss_lagrange": constraint_loss.detach().item()
        }
        return constraint_loss, metrics

    def get_sigmoid_lagrange(self, detach=False):
        if detach:
            return torch.sigmoid(self.lagrange_parameter.detach()).to(self.cfg.training.device)
        else:
            return torch.sigmoid(self.lagrange_parameter).to(self.cfg.training.device)

    def postprocess_clip_lagrange(self):
        # Keep the lagrange parameter in a tractable range: -15, 15
        self.lagrange_parameter.data = torch.clamp(self.lagrange_parameter.data, -15., 15.)

    def update_lagrange(self, loss, postprocess_clip=True):
        self.lagrange_optimizer.zero_grad()
        loss.backward()
        self.lagrange_optimizer.step()
        if postprocess_clip:
            self.postprocess_clip_lagrange()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
