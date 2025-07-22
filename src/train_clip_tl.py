"""Train a CLIP Tuned Lens on a dataset."""

import json
import logging
from typing import Union
import torch
import torch.nn.functional as F
import wandb
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import open_clip

from src.clip_tl import CLIPTunedLens, get_clip_hidden_states
from src.ingredients import CLIPModel, ImageData, Optimizer
from src.utils import plot_training_history

logger = logging.getLogger(__name__)


LossChoice = ["MSE", "KL"]


class Train:
    """Train a CLIP Tuned Lens on a dataset.

    The training process is controlled by the following parameters in configs:
    - num_epochs: The number of epochs to train for.
    - validate_every_n_steps: The number of steps between validation.
    - loss_function: The loss function to use.
    - early_stopping: The early stopping configuration.
    - output: The output directory.
    - scheduler: The scheduler to use.

    We use num_epochs for controlling the total number of data seen, and
    global_step for controlling the number of steps. Val loss is calculated
    every self.validate_every_n_steps steps.

    Args:
        config: The full training configuration dictionary from YAML.
        model: The CLIP model configuration to use.
        data: The image data configuration.
        optimizer: The optimizer configuration.
    """

    def __init__(self,
                 config: dict,
                 model: CLIPModel,
                 data: ImageData,
                 optimizer: Optimizer,
                 project_root: Path):
        """Initialize the trainer.

        Args:
            config: The full training configuration dictionary from YAML.
            model: The CLIP model configuration to use.
            data: The image data configuration.
            optimizer: The optimizer configuration.
            project_root: The root directory of the project.
        """
        self.config = config
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.project_root = project_root

        # Extract frequently used configs
        self.training_cfg = self.config.get('training', {})
        self.output_cfg = self.config.get('output', {})
        exp_name = self.output_cfg.get('experiment_name')
        if not exp_name:
            exp_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_name = exp_name
        # All output paths are now relative to the project root.
        self.output_dir = self.project_root / self.output_cfg.get(
            'base_dir', 'outputs') / self.exp_name

        self.seed = self.training_cfg.get('seed', 42)
        self.num_epochs = self.training_cfg.get('num_epochs', 20)
        self.validate_every_n_steps = self.training_cfg.get(
            'validate_every_n_steps', 100)
        self.loss_fn_name = self.training_cfg.get('loss_function', 'MSE')
        assert self.loss_fn_name in LossChoice, \
            f'Unknown loss function: {self.loss_fn_name}'
        self.early_stopping_metric = self.training_cfg.get(
            'early_stopping', {}).get('metric', 'val_loss')

        # Set checkpoint directory default
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.use_wandb = self.config.get('logging', {}).get('use_wandb', False)

    def setup(self):
        """Load model, data, lens, optimizer, and scheduler(if any)."""

        # load model (CLIPModel will automatically extract config)
        model, preprocess = self.model.load()
        if preprocess is None:
            raise ValueError('Preprocessing function not loaded from model.')
        train_loader, val_loader = self.data.load(preprocess)

        # get all parameters that require gradients
        # if self.lens_path is None:
        lens = CLIPTunedLens.from_clip_model(self.model)

        # Move lens to the same device as the model
        lens = lens.to(self.model.device)

        params = [p for p in lens.parameters() if p.requires_grad]
        optimizer = self.optimizer.create_optim(params)

        # --- Calculate total training steps ---
        total_training_steps = len(train_loader) * self.num_epochs

        # --- Create Learning Rate Scheduler ---
        scheduler_cfg = self.config.get('scheduler', {})
        scheduler_name = scheduler_cfg.get('name')
        scheduler = None

        if scheduler_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_cfg.get('step_size', 10),
                gamma=scheduler_cfg.get('gamma', 0.1)
            )
            logger.info('Using StepLR scheduler.')

        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_cfg.get('T_max', total_training_steps)
            )
            logger.info(f'Using CosineAnnealingLR scheduler with T_max = '
                        f'{total_training_steps}.')

        else:
            logger.info('No learning rate scheduler used.')

        assert model and preprocess and train_loader and lens and optimizer

        return model, train_loader, val_loader, optimizer, lens, scheduler

    def run(self) -> None:
        """Main training loop.

        In this loop, we use epoch for controling the total number of data
        seen, and global_step for controling the number of steps. Val loss
        is calculated every self.validate_every_n_steps steps.
        """

        # --- WandB ---
        if self.use_wandb:
            wandb.init(
                project="clip-tl",
                name=self.exp_name,
                config=self.config,
                dir=str(self.output_dir),  # wandb needs a string path
            )

        # --- Setup ---
        (
            model,
            train_loader,
            val_loader,
            optimizer,
            lens,
            scheduler,
        ) = self.setup()

        # --- Training State Tracking ---
        if self.use_wandb:
            wandb.watch(lens, log="all", log_freq=self.validate_every_n_steps)

        # --- Early Stopping Initialization ---
        early_stop_cfg = self.training_cfg.get('early_stopping', {})
        early_stopping_enabled = early_stop_cfg.get('enabled', False)
        patience = early_stop_cfg.get('patience', 10)
        best_loss = float('inf')
        patience_counter = 0

        # --- Training Loop ---
        total_training_steps = len(train_loader) * self.num_epochs
        lens.train()
        pbar = tqdm(total=total_training_steps, desc='Training')

        # --- Training State Tracking ---
        training_history = []
        global_step = 0
        train_loss_accumulator = 0.0
        steps_since_last_val = 0

        # --- Training Loop ---
        for epoch in range(self.num_epochs):
            for batch_idx, (images, _) in enumerate(train_loader):
                epoch_step = batch_idx + 1

                images = images.to(self.model.device)
                with get_clip_hidden_states(model, images) as (final_logits,
                                                               hidden_states):
                    batch_loss = 0.0

                    for layer_idx, h in enumerate(hidden_states[:-1]):
                        layer_output = lens(h, layer_idx)
                        if self.loss_fn_name == 'MSE':
                            layer_loss = F.mse_loss(layer_output, final_logits)
                        elif self.loss_fn_name == 'KL':
                            layer_p = layer_output.float().log_softmax(dim=-1)
                            final_p = final_logits.float().log_softmax(dim=-1)
                            layer_loss = F.kl_div(layer_p,
                                                  final_p,
                                                  reduction='batchmean')
                        else:
                            raise ValueError(f'Unknown loss function: '
                                             f'{self.loss_fn_name}')

                        layer_loss.backward()

                        batch_loss += layer_loss.item()

                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler:
                        scheduler.step()

                    train_loss_accumulator += batch_loss
                    global_step += 1
                    steps_since_last_val += 1
                    pbar.update(1)

                    # --- Log step-wise training info ---
                    current_lr = optimizer.param_groups[0]['lr']
                    if self.use_wandb:
                        wandb.log({
                            'train/step_loss': batch_loss,
                            'train/lr': current_lr,
                            'epoch': epoch + 1,
                        }, step=global_step)  # use global_step as x-axis

                    training_history.append({
                        'step': global_step,
                        'epoch': epoch + 1,
                        'train_loss': batch_loss,
                        'lr': current_lr,
                    })

                # update pbar for each step
                pbar.set_postfix({
                    'Epoch': f'{epoch + 1}/{self.num_epochs}',
                    'Step Loss': f'{batch_loss:.4f}',
                    'Avg Loss': f'{train_loss_accumulator / global_step:.4f}'
                })

                # --- Validation, Logging, and Checkpointing ---
                if global_step % self.validate_every_n_steps == 0:
                    avg_train_loss = (train_loss_accumulator /
                                      steps_since_last_val)
                    current_lr = optimizer.param_groups[0]['lr']

                    log_msg = (
                        f'Step {global_step}/{total_training_steps} '
                        f'(in Epoch {epoch + 1}) finished, '
                        f'avg train loss (last {steps_since_last_val} steps): '
                        f'{avg_train_loss:.4f}, '
                        f'lr: {current_lr:.6f}'
                    )

                    # --- Validation Step ---
                    avg_val_loss = None
                    if val_loader:
                        avg_val_loss = self.run_validation(lens,
                                                           model, val_loader)
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

                    # --- Early stopping check ---
                    if early_stopping_enabled:
                        monitored_loss = None
                        # Determine which loss to monitor based on config
                        if self.early_stopping_metric == 'val_loss':
                            if val_loader and avg_val_loss is not None:
                                monitored_loss = avg_val_loss
                                metrics = {
                                    'avg_train_loss': avg_train_loss,
                                    'avg_val_loss': avg_val_loss
                                }
                            else:
                                # use train_loss if val_loss is not available
                                monitored_loss = avg_train_loss
                                logger.warning("val_loss not available, "
                                               "use train_loss for early "
                                               "stopping.")
                                metrics = {
                                    'avg_train_loss': avg_train_loss,
                                }
                        else:  # metric is 'train_loss'
                            monitored_loss = avg_train_loss
                            metrics = {
                                'avg_train_loss': avg_train_loss,
                            }

                        if monitored_loss < best_loss:
                            best_loss = monitored_loss
                            patience_counter = 0
                            # Save the best model
                            self.save_checkpoint(
                                epoch=epoch + 1,
                                epoch_step=epoch_step,
                                global_step=global_step,
                                lens=lens,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                metrics=metrics,
                                use_wandb=self.use_wandb,
                                is_best=True)
                        else:
                            patience_counter += 1
                            logger.warning(f'Early stopping counter: '
                                           f'{patience_counter}/{patience}')
                            if patience_counter >= patience:
                                logger.info('Early stopping triggered. '
                                            'Stopping training.')
                                pbar.close()
                                return  # Exit training function

                        # --- Save regular checkpoint ---
                        self.save_checkpoint(
                            epoch=epoch + 1,
                            epoch_step=epoch_step,
                            global_step=global_step,
                            lens=lens,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            metrics=metrics,
                            use_wandb=self.use_wandb,
                            is_best=False)

        pbar.close()

        # save training history and plot
        training_history_path = self.output_dir / 'training_history.json'
        with open(training_history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info('Training history saved to '
                    f'{training_history_path.as_posix()}')

        # Plot training history
        fig_step, fig_avg = plot_training_history(
            history=training_history,
            output_dir=self.output_dir,
            exp_name=self.exp_name
        )

        if self.use_wandb:
            wandb.log({
                "charts/step_loss": wandb.Plotly(fig_step),
                "charts/avg_loss_lr": wandb.Plotly(fig_avg),
            })
            wandb.finish()

        logger.info(f'Training finished after {self.num_epochs} epochs.')

    def run_validation(self, lens: CLIPTunedLens, model: open_clip.model.CLIP,
                       val_loader: DataLoader) -> float:
        """Run validation and return the average validation loss.
        """
        lens.eval()
        total_val_loss = 0.0

        pbar = tqdm(val_loader, desc='Validating')
        with torch.no_grad():
            for images, _ in pbar:
                images = images.to(self.model.device)

                with get_clip_hidden_states(model, images) as \
                        (final_logits, hidden_states):

                    batch_loss = 0.0
                    for layer_idx, h in enumerate(hidden_states[:-1]):
                        layer_output = lens(h, layer_idx)
                        if self.loss_fn_name == 'MSE':
                            layer_loss = F.mse_loss(layer_output, final_logits)
                        elif self.loss_fn_name == 'KL':
                            layer_p = layer_output.float().log_softmax(dim=-1)
                            final_p = final_logits.float().log_softmax(dim=-1)
                            layer_loss = F.kl_div(layer_p, final_p,
                                                  reduction='batchmean')
                        else:
                            raise ValueError(f'Unknown loss function: '
                                             f'{self.loss_fn_name}')
                        batch_loss += layer_loss.item()
                    total_val_loss += batch_loss

                pbar.set_postfix({
                    'Batch Val Loss': f'{batch_loss:.4f}',
                    'Avg Val Loss': f'{total_val_loss / (pbar.n + 1):.4f}'
                })

        return total_val_loss / len(val_loader)

    def save_checkpoint(self,
                        epoch: int,
                        epoch_step: int,
                        global_step: int,
                        lens: CLIPTunedLens,
                        optimizer: torch.optim.Optimizer,
                        scheduler: Union[
                            torch.optim.lr_scheduler.StepLR,
                            torch.optim.lr_scheduler.CosineAnnealingLR,
                            None],
                        metrics: dict,
                        use_wandb: bool,
                        is_best: bool = False) -> None:
        """Save the checkpoint."""

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
            logger.info(f'Saving best model checkpoint at epoch {epoch}, '
                        f'epoch_step {epoch_step}, global_step {global_step} '
                        f'with metrics {metrics}')
            if use_wandb and wandb.run:
                # create wandb artifact
                artifact = wandb.Artifact(
                    name=f'{wandb.run.name}_best_lens',
                    type='model',
                    metadata={'epoch': epoch,
                              'epoch_step': epoch_step,
                              'global_step': global_step,
                              'metrics': metrics})
                artifact.add_file(checkpoint_path.as_posix())
                wandb.log_artifact(artifact, aliases=['best', 'latest'])
        else:
            checkpoint_path = (self.checkpoint_dir /
                               f'checkpoint_e{epoch}_gs{global_step}.pth')
        torch.save({
            'epoch': epoch,
            'epoch_step': epoch_step,
            'global_step': global_step,
            'lens_state_dict': lens.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': (scheduler.state_dict()
                                     if scheduler is not None else None),
            'metrics': metrics,
        }, checkpoint_path)
        logger.info(f'Checkpoint saved to {checkpoint_path.as_posix()}')
