"""Train a CLIP Tuned Lens on a dataset."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.train.utils import (
    load_config,
    plot_training_history,
    resolve_path_from_config,
    setup_logging,
)
from src.tuned_lens.clip_tl import CLIPTunedLens, get_clip_hidden_states
from src.tuned_lens.ingredients import CLIPModel, ImageData, Optimizer

logger = logging.getLogger(__name__)


LossChoice = ['MSE', 'KL']


class Train:
    """Train a CLIP Tuned Lens on a dataset.

    We use num_epochs for controlling the total number of data seen, and
    global_step for controlling the number of steps. Val loss is calculated
    every self.validate_every_n_steps steps.

    Args:
        config: The full training configuration dictionary from YAML.
        project_root: The root directory of the project.
        output_dir: The output directory.
    """

    def __init__(self, config: dict, project_root: Path, output_dir: Path):
        """Initialize the trainer.

        Args:
            config: The full training configuration dictionary from YAML.
            project_root: The root directory of the project.
            output_dir: The output directory.
        """

        self.config = config
        self.project_root = project_root
        self.output_dir = output_dir

        # --- Extract frequently used configs ---
        self.training_cfg = self.config.get('training', {})
        self.output_cfg = self.config.get('output', {})
        exp_name = self.output_cfg.get('experiment_name')
        if not exp_name:
            exp_name = f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.exp_name = exp_name

        # --- Setup device ---
        self.device = self._setup_device()

        # --- Training configs ---
        self.seed = self.training_cfg.get('seed', 42)
        self.num_epochs = self.training_cfg.get('num_epochs', 20)
        self.validate_every_n_steps = self.training_cfg.get(
            'validate_every_n_steps', 100
        )
        self.loss_fn_name = self.training_cfg.get('loss_function', 'MSE')
        assert self.loss_fn_name in LossChoice, (
            f'Unknown loss function: {self.loss_fn_name}'
        )

        # --- Checkpoint and early stopping ---
        self.early_stopping_metric = self.training_cfg.get(
            'early_stopping', {}
        ).get('metric', 'val_loss')
        self.checkpoint_every = self.training_cfg.get('checkpoint_every', 500)
        self.checkpoint_dir = (
            resolve_path_from_config(
                self.training_cfg.get('checkpoint_dir'), self.project_root
            )
            if self.training_cfg.get('checkpoint_dir') is not None
            else self.output_dir / 'checkpoints'
        )

        # --- Logging ---
        self.use_wandb = self.config.get('logging', {}).get('use_wandb', False)

        # --- Resume from checkpoint ---
        self.resume_from_checkpoint = self.training_cfg.get(
            'resume_from_checkpoint', None
        )

        # --- Setup components ---
        self._setup_components()

    def _setup_components(self):
        """Load model, data, lens, optimizer, and scheduler(if any)."""

        self.model = self._setup_model()
        # will also setup model.model, model.preprocess
        if self.model.preprocess is None:
            raise ValueError('Preprocessing function not loaded from model.')
        self.train_loader, self.val_loader = self._setup_dataloader(
            self.model.preprocess
        )

        self.lens = self._setup_lens(self.model)
        params = [p for p in self.lens.parameters() if p.requires_grad]
        self.optimizer = self._setup_optimizer(params)

        total_training_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = self._setup_scheduler(
            self.optimizer, total_training_steps
        )

        assert (
            self.model
            and self.train_loader
            and self.lens
            and self.optimizer
            and self.scheduler
        )

    def _setup_device(self):
        device = self.training_cfg.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _setup_model(self):
        model = CLIPModel(
            model_name=self.config['model']['name'], device=self.device
        )
        model.load()
        return model

    def _setup_dataloader(
        self, preprocess: Callable
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        data_path = resolve_path_from_config(
            self.config['data']['path'], self.project_root
        )
        data = ImageData(
            data_path=str(data_path),  # ImageData expects a string path
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            validation_split=self.config['data'].get('validation_split', 0.0),
        )
        return data.load(preprocess)  # return train_loader, val_loader(if any)

    def _setup_lens(self, model: CLIPModel):
        lens = CLIPTunedLens.from_clip_model(model)
        lens = lens.to(self.device)
        return lens

    def _setup_optimizer(
        self, params: list[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        optimizer = Optimizer(
            optimizer=self.config['optimizer']['name'],
            lr=self.config['optimizer']['lr'],
            weight_decay=self.config['optimizer']['weight_decay'],
            beta1=self.config['optimizer'].get('beta1', 0.9),
            beta2=self.config['optimizer'].get('beta2', 0.999),
            momentum=self.config['optimizer'].get('momentum', 0.9),
        )
        return optimizer.create_optim(params)

    def _setup_scheduler(
        self, optimizer: torch.optim.Optimizer, total_training_steps: int
    ) -> Union[
        torch.optim.lr_scheduler.StepLR,
        torch.optim.lr_scheduler.CosineAnnealingLR,
        torch.optim.lr_scheduler.SequentialLR,
        None,
    ]:
        scheduler_cfg = self.config.get('scheduler', {})
        scheduler_name = scheduler_cfg.get('name')
        use_warmup = scheduler_cfg.get('warmup', {}).get('enabled', False)

        if not scheduler_name:
            logger.info('No learning rate scheduler used.')
            return None

        if scheduler_name == 'StepLR':
            logger.info('Using StepLR scheduler.')
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_cfg.get('step_size', 10),
                gamma=scheduler_cfg.get('gamma', 0.1),
            )
        elif scheduler_name == 'CosineAnnealingLR':
            logger.info(
                f'Using CosineAnnealingLR scheduler with T_max = '
                f'{total_training_steps}.'
            )
            t_max = total_training_steps

            if use_warmup:
                t_max -= scheduler_cfg.get('warmup', {}).get('steps', 500)
                if t_max <= 0:
                    raise ValueError('T_max must be greater than 0.')
                logger.info(
                    f'Using warmup with '
                    f'{scheduler_cfg.get("warmup", {}).get("steps", 500)} '
                    f'steps.'
                )

            logger.info(
                f'Using CosineAnnealingLR scheduler with T_max = {t_max}.'
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max
            )
        else:
            logger.info('No learning rate scheduler used.')

        # --- combine warmup and cosine annealing ---
        if use_warmup:
            warmup_steps = scheduler_cfg.get('warmup', {}).get('steps', 500)
            warmup_start_factor = scheduler_cfg.get('warmup', {}).get(
                'start_factor', 0.001
            )
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_steps],
            )

        return scheduler

    def _load_from_checkpoint(self):
        """Load from checkpoint and return the start epoch and global step.

        Returns:
            start_epoch: The epoch to start from.
            global_step: The global step to start from.
            best_loss: The best loss so far.
        """
        logger.info(f'Resuming from checkpoint:{self.resume_from_checkpoint}')
        if not Path(self.resume_from_checkpoint).exists():
            raise FileNotFoundError(
                f'Checkpoint file not found: {self.resume_from_checkpoint}'
            )
        checkpoint = torch.load(self.resume_from_checkpoint)
        self.lens.load_state_dict(checkpoint['lens_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_loss = checkpoint.get('metrics', {}).get(
            'avg_val_loss', float('inf')
        )
        logger.info(
            f'Resumed from epoch {start_epoch}, '
            f'global step {global_step}, best loss {best_loss}'
        )
        return start_epoch, global_step, best_loss

    def run(self) -> None:
        """Main training loop.

        In this loop, we use epoch for controling the total number of data
        seen, and global_step for controling the number of steps. Val loss
        is calculated every self.validate_every_n_steps steps.
        """

        # --- Setup ---
        start_epoch = 0
        global_step = 0
        best_loss = float('inf')

        if self.resume_from_checkpoint:  # replace lens, optimizer, scheduler
            start_epoch, global_step, best_loss = self._load_from_checkpoint()

        # --- WandB ---
        if self.use_wandb:
            wandb.watch(
                self.lens, log='all', log_freq=self.validate_every_n_steps
            )

        # --- Early Stopping Initialization ---
        early_stop_cfg = self.training_cfg.get('early_stopping', {})
        early_stopping_enabled = early_stop_cfg.get('enabled', False)
        patience = early_stop_cfg.get('patience', 10)
        best_loss = float('inf')
        patience_counter = 0

        # --- Training State Tracking ---
        training_history = []
        train_loss_accumulator = 0.0
        steps_since_last_val = 0
        training_stop = False  # flag for nested training loop

        # --- Training Loop ---
        total_training_steps = len(self.train_loader) * self.num_epochs
        self.lens.train()
        pbar = tqdm(total=total_training_steps, desc='Training')

        for epoch in range(start_epoch, self.num_epochs):
            for batch_idx, (images, _) in enumerate(self.train_loader):
                epoch_step = batch_idx + 1
                batch_loss = self._run_batch(images, is_training=True)

                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

                train_loss_accumulator += batch_loss
                global_step += 1
                steps_since_last_val += 1
                pbar.update(1)

                # --- Log step-wise training info ---
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.use_wandb:
                    wandb.log(
                        {
                            'train/step_loss': batch_loss,
                            'train/lr': current_lr,
                            'epoch': epoch + 1,
                        },
                        step=global_step,
                    )

                training_history.append(
                    {
                        'step': global_step,
                        'epoch': epoch + 1,
                        'train_loss': batch_loss,
                        'lr': current_lr,
                    }
                )

                # update pbar for each step
                pbar.set_postfix(
                    {
                        'Epoch': f'{epoch + 1}/{self.num_epochs}',
                        'Step Loss': f'{batch_loss:.4f}',
                        'Avg Loss': f'{
                            train_loss_accumulator / steps_since_last_val:.4f}',
                    }
                )

                # --- Validation, Logging, and Checkpointing ---
                if global_step % self.validate_every_n_steps == 0:
                    avg_train_loss = (
                        train_loss_accumulator / steps_since_last_val
                    )
                    current_lr = self.optimizer.param_groups[0]['lr']

                    log_msg = (
                        f'Step {global_step}/{total_training_steps} '
                        f'(in Epoch {epoch + 1}) finished, '
                        f'avg train loss (last {steps_since_last_val} steps): '
                        f'{avg_train_loss:.4f}, '
                        f'lr: {current_lr:.6f}'
                    )

                    # --- Validation Step ---
                    avg_val_loss = None
                    if self.val_loader:
                        avg_val_loss = self.run_validation()
                        log_msg += f', avg val loss: {avg_val_loss:.4f}'

                    logger.info(log_msg)

                    # --- Log aggregated metrics ---
                    # The last item in history corresponds to the current step
                    log_dict = {
                        'train/avg_loss': avg_train_loss,
                    }
                    if avg_val_loss is not None:
                        log_dict['val/avg_loss'] = avg_val_loss

                    if self.use_wandb:
                        wandb.log(log_dict, step=global_step)

                    history_log = {
                        'step': global_step,
                        'epoch': epoch + 1,
                        'avg_train_loss': avg_train_loss,
                        'lr': current_lr,
                    }
                    if avg_val_loss is not None:
                        history_log['avg_val_loss'] = avg_val_loss

                    training_history.append(history_log)

                    # Reset accumulators for next validation cycle
                    train_loss_accumulator = 0.0
                    steps_since_last_val = 0

                    monitored_loss = (
                        avg_val_loss
                        if avg_val_loss is not None
                        else avg_train_loss
                    )
                    metrics = {
                        'avg_train_loss': avg_train_loss,
                    }
                    if avg_val_loss is not None:
                        metrics['avg_val_loss'] = avg_val_loss

                    # --- Save best model ---
                    if monitored_loss < best_loss:
                        best_loss = monitored_loss
                        patience_counter = 0
                        self.save_checkpoint(
                            epoch=epoch,
                            epoch_step=epoch_step,
                            global_step=global_step,
                            metrics=metrics,
                            is_best=True,
                        )

                    # --- Early stopping check ---
                    elif early_stopping_enabled:
                        patience_counter += 1
                        logger.warning(
                            f'Early stopping counter: '
                            f'{patience_counter}/{patience}'
                        )
                        if patience_counter >= patience:
                            logger.info(
                                'Early stopping triggered. Stopping training.'
                            )
                            pbar.close()
                            training_stop = True

                    # --- Save regular checkpoint ---
                    if global_step % self.checkpoint_every == 0:
                        self.save_checkpoint(
                            epoch=epoch,
                            epoch_step=epoch_step,
                            global_step=global_step,
                            metrics=metrics,
                            is_best=False,
                        )

                if training_stop:  # batch-wise training stop
                    break

            if training_stop:  # epoch-wise training stop
                break

        pbar.close()

        # --- Save training history and plot ---
        training_history_path = self.output_dir / 'training_history.json'
        with open(training_history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(
            f'Training history saved to {training_history_path.as_posix()}'
        )

        # --- Plot training history ---
        fig_step, fig_avg = plot_training_history(
            history=training_history,
            output_dir=self.output_dir,
            exp_name=self.exp_name,
        )

        if self.use_wandb:
            # wandb.log({
            #     "charts/step_loss": wandb.Plotly(fig_step),
            #     "charts/avg_loss_lr": wandb.Plotly(fig_avg),
            # })
            wandb.finish()

        logger.info(f'Training finished after {self.num_epochs} epochs.')

        # --- Save final model from best checkpoint ---
        best_checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        if best_checkpoint_path.exists():
            logger.info(
                'Loaded best model from '
                f'{best_checkpoint_path.as_posix()}'
                'to save final lens.'
            )
            checkpoint = torch.load(best_checkpoint_path)
            self.lens.load_state_dict(checkpoint['lens_state_dict'])

            final_lens_dir = self.output_dir / 'final_lens'
            logger.info(f'Saving final lens to {final_lens_dir.as_posix()}')
            self.lens.save(final_lens_dir)
        else:
            logger.warning(
                'No best model found. The last state of the model '
                'will not be saved.'
            )

    def run_validation(self) -> float:
        """Run validation and return the average validation loss."""
        self.lens.eval()
        total_val_loss = 0.0
        assert self.val_loader and self.lens and self.model.model

        pbar = tqdm(self.val_loader, desc='Validating')
        with torch.no_grad():
            for images, _ in pbar:
                batch_loss = self._run_batch(images, is_training=False)
                total_val_loss += batch_loss

                pbar.set_postfix(
                    {
                        'Batch Val Loss': f'{batch_loss:.4f}',
                        'Avg Val Loss': f'{total_val_loss / (pbar.n + 1):.4f}',
                    }
                )

        return total_val_loss / len(self.val_loader)

    def _run_batch(self, images: torch.Tensor, is_training: bool) -> float:
        """Run a batch of train / val, return the batch loss (only value)."""
        images = images.to(self.device)
        assert self.model.model and self.lens

        batch_loss = 0.0
        with get_clip_hidden_states(self.model.model, images) as (
            final_logits,
            hidden_states,
        ):
            for layer_idx, h in enumerate(hidden_states[:-1]):
                layer_output = self.lens(h, layer_idx)
                if self.loss_fn_name == 'MSE':
                    layer_loss = F.mse_loss(layer_output, final_logits)
                elif self.loss_fn_name == 'KL':
                    layer_p = layer_output.float().log_softmax(dim=-1)
                    final_p = final_logits.float().log_softmax(dim=-1)
                    layer_loss = F.kl_div(
                        layer_p, final_p, reduction='batchmean'
                    )
                else:
                    raise ValueError(
                        f'Unknown loss function: {self.loss_fn_name}'
                    )
                if is_training:
                    layer_loss.backward()

                batch_loss += layer_loss.item()

            return batch_loss

    def save_checkpoint(
        self,
        epoch: int,
        epoch_step: int,
        global_step: int,
        metrics: dict,
        is_best: bool = False,
    ) -> None:
        """Save the checkpoint."""

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            logger.info(
                f'Saving best model checkpoint at epoch {epoch}, '
                f'epoch_step {epoch_step}, global_step {global_step} '
                f'with metrics {metrics}.'
            )
        else:
            checkpoint_path = (
                self.checkpoint_dir / f'checkpoint_e{epoch}_gs{global_step}.pth'
            )
        torch.save(
            {
                'epoch': epoch,
                'epoch_step': epoch_step,
                'global_step': global_step,
                'lens_state_dict': self.lens.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),
                'metrics': metrics,
            },
            checkpoint_path,
        )
        logger.info(f'Checkpoint saved to {checkpoint_path.as_posix()}.')

        # # --- Save best model to wandb ---
        # if self.use_wandb and wandb.run and is_best:
        #     logger.info('Saving best model to wandb.')
        #     artifact = wandb.Artifact(
        #         name=f'{wandb.run.name}_best_lens',
        #         type='model',
        #         metadata={'epoch': epoch,
        #                   'epoch_step': epoch_step,
        #                   'global_step': global_step,
        #                   'metrics': metrics})
        #     artifact.add_file(checkpoint_path.as_posix())
        #     wandb.log_artifact(artifact, aliases=['best', 'latest'])


def main(project_root: Path, config_path: Path):
    """Main training function."""

    # --- Resolve Paths ---
    # Config path is relative to the project root, or use absolute path
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.is_file():
        raise FileNotFoundError(
            f'Configuration file not found at: {config_path}'
        )

    # Load config from YAML
    config = load_config(config_path)

    # --- Setup Output Directory ---
    output_cfg = config.get('output', {})
    exp_name = output_cfg.get('experiment_name')
    if not exp_name:
        exp_name = f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # Output base directory is relative to the project root.
    base_dir = resolve_path_from_config(
        output_cfg.get('base_dir', 'outputs'), project_root
    )
    output_dir = base_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup Logging ---
    log_cfg = config.get('logging', {})
    log_path = output_dir / 'training.log'
    setup_logging(log_path, log_cfg.get('level', 'INFO'))

    logger = logging.getLogger(__name__)
    logger.info('Starting training with config: %s', config_path)
    logger.info('Output directory: %s', output_dir)
    logger.info('Log file: %s', log_path)

    # --- Setup WandB ---
    def update_config(original_config, sweep_params):
        for key, value in sweep_params.items():
            if '.' in key:
                keys = key.split('.')
                d = original_config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
            else:
                original_config[key] = value

    use_wandb = log_cfg.get('use_wandb', False)
    logger.info(f'use_wandb: {use_wandb}')
    if use_wandb:
        wandb.init(
            project='clip-tl',
            name=config.get('experiment_name'),
            config=config,
            dir=str(config.get('output_dir')),
        )
        sweep_config = wandb.config
        update_config(config, sweep_config)

    # Save the config for this run for reproducibility
    shutil.copyfile(config_path, output_dir / 'config.yaml')
    logger.info('Saved config snapshot to %s', output_dir / 'config.yaml')

    # --- Create trainer ---
    trainer = Train(
        config=config, output_dir=output_dir, project_root=project_root
    )

    try:
        logger.info('Starting training...')
        trainer.run()
        logger.info('Training completed successfully.')
        logger.info(f'All training artifacts saved to: {output_dir}.')

    except Exception as e:
        logger.error(f'Training failed with error: {e}', exc_info=True)
        raise

    logger.info('All done.')
