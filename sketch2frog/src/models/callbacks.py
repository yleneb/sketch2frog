import gc
from numpy import geomspace
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from src.visualisation.visualisation import show_single_prediction
from src.visualisation.visualisation import show_batch_prediction

class SinglePredictionCallback(Callback):
    def __init__(self, steps_per_epoch, sample_sketch, sample_image, valid_ds, continue_training=False, patch_cmap='viridis'):
        """
        During the first epoch we want to log more frequently.
        If we continue training from the nth epoch we don't want 
        these extra logs on epoch n.
        
        During the first epoch, logs are made evenly on a logarithmic scale."""
        super(SinglePredictionCallback, self).__init__()
        self.sample_sketch = sample_sketch
        self.sample_image = sample_image
        self.valid_ds = valid_ds
        self.is_first_epoch = not(continue_training)
        self.batches_to_log = geomspace(1, steps_per_epoch, 10).astype(int) - 1
        self.patch_cmap = patch_cmap
        
    def on_epoch_end(self, epoch, logs=None):
        show_single_prediction(self.model, self.sample_sketch, self.sample_image,
            wandb_title=f'Training example', patch_cmap=self.patch_cmap)
        
        # plot validation example
        for x_batch, y_batch in self.valid_ds.take(1):
            val_sketch, val_image = x_batch[0], y_batch[0]
            show_single_prediction(self.model, val_sketch, val_image,
                wandb_title=f'Validation example', patch_cmap=self.patch_cmap)

    def on_train_begin(self, epoch, logs=None):
        # only at the very start of training.
        # Not if we are continuing training
        if self.is_first_epoch:
            show_single_prediction(self.model, self.sample_sketch, self.sample_image,
                wandb_title='Training example before training', patch_cmap=self.patch_cmap)
    
    def on_train_batch_end(self, batch, logs=None):
        # we only want to log during the first epoch
        if self.is_first_epoch:
            # only log specific batches, spaced on a log curve
            if batch in self.batches_to_log:
                show_single_prediction(self.model, self.sample_sketch, self.sample_image,
                    wandb_title=f'Training example Epoch 0', patch_cmap=self.patch_cmap)
                
                # plot validation example
                for x_batch, y_batch in self.valid_ds.take(1):
                    val_sketch, val_image = x_batch[0], y_batch[0]
                    show_single_prediction(self.model, val_sketch, val_image,
                        wandb_title=f'Validation example Epoch 0', patch_cmap=self.patch_cmap)
                
                # after saving examples of the last batch in epoch one
                # set is_first_batch to false
                if batch == self.batches_to_log[-1]:
                    self.is_first_epoch = False
                    
class BatchPredictionCallback(Callback):
    def __init__(self, sketches, steps_per_epoch, continue_training=False):
        """
        During the first epoch we want to log more frequently.
        If we continue training from the nth epoch we don't want 
        these extra logs on epoch n.
        
        During the first epoch, logs are made evenly on a logarithmic scale."""
        super(BatchPredictionCallback, self).__init__()
        self.is_first_epoch = not(continue_training)
        self.batches_to_log = geomspace(1, steps_per_epoch, 10).astype(int) - 1
        self.sketches = sketches
        
    def on_epoch_end(self, epoch, logs=None):      
        show_batch_prediction(self.model, self.sketches, wandb_title='Sample Batch')

    def on_train_begin(self, epoch, logs=None):
        # only at the very start of training.
        if self.is_first_epoch: # Not if we are continuing training
            show_batch_prediction(self.model, self.sketches, wandb_title='Sample Batch Before Training')
    
    def on_train_batch_end(self, batch, logs=None):
        # we only want to log during the first epoch
        if self.is_first_epoch:
            # only log specific batches, spaced on a log curve
            if batch in self.batches_to_log:
                show_batch_prediction(self.model, self.sketches, wandb_title='Sample Batch Epoch 0')
                
                # after saving examples of the last batch in epoch one
                # set is_first_batch to false
                if batch == self.batches_to_log[-1]:
                    self.is_first_epoch = False
                    
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()