import os
import wandb
from glob import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from IPython.display import display
from matplotlib import cm
import matplotlib.cbook
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tensorflow import Tensor, newaxis, ones_like
from tensorflow.keras.preprocessing.image import array_to_img
from src.tools.tools import ceildiv

def viridis_extended():
    """Returns an extended viridis colormap for PatchGAN outputs
           x < -0.5  black
    -0.5 < x <  0    black -> viridis purple
     0   < x <  1    iridis
     1   < x <  1.5  viridis yellow -> white
     1.5 < x         white"""
  
    # get viridis and its reverse for convenience
    viridis = cm.get_cmap('viridis', 128)
    viridis_r = cm.get_cmap('viridis_r', 128)

    # left side: black -> viridis purple
    left_nodes = [0, 1]
    left_colors = [(0,0,0), viridis(0)[:3]]
    left = LinearSegmentedColormap.from_list("black_blue", list(zip(left_nodes, left_colors)))

    # right side viridis yellow -> white
    right_nodes = [0,1]
    right_colors = [viridis_r(0)[:3], (1,1,1)]
    right = LinearSegmentedColormap.from_list("black_blue", list(zip(right_nodes, right_colors)))

    viridis_extended = np.vstack((
        left(np.linspace(0, 1, 50)),
        viridis(np.linspace(0, 1, 100)),
        right(np.linspace(0, 1, 50))))

    return ListedColormap(viridis_extended)

def show_batch(dataset, size=3, max_examples=8):
    """Print a batch of sketches and their target frogs,
    for large batch sizes limit number to print"""
    
    # get a batch and prep the figure
    for sketch_batch, image_batch in dataset.take(1):
        print(sketch_batch.shape, image_batch.shape)
        # determine how many images to plot
        to_plot = min(sketch_batch.shape[0], max_examples)
        fig, axs = plt.subplots(2, to_plot, figsize=(size*to_plot, size*2))
        
        # print each img in the batch
        for i, (sketch, image) in enumerate(zip(sketch_batch, image_batch)):
            axs[0,i].imshow(sketch.numpy().squeeze(), cmap='gray')
            axs[1,i].imshow((image.numpy().squeeze()+1)/2)
            if i == to_plot-1:
                break

    # remove axis ticks
    for ax in matplotlib.cbook.flatten(axs):
        ax.set(xticks=[], yticks=[])

    # set titles
    axs[0,0].set_ylabel('Sketch')
    axs[1,0].set_ylabel('Source Image')

    # adjust styling and show
    fig.patch.set_facecolor('white')      
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
    plt.close()
    
def save_sample(dataset, save_dir, batch_size=16, use_wandb=False):
    """Prepares 16 images from the validation set to use for visualation
    during training. Saves in a temporary folder if needed to continuing learning.
    Optionally saves image of sketches and targets to wandb."""
    # take a 16 examples from the valid_ds to use for visualisation
    sample_ds = (dataset
        .take(ceildiv(16, batch_size)) # take enough batches
        .unbatch() # split up
        .take(16) # take enough examples
        .batch(16)) # make into 1 batch

    # in wandb
    SIZE = 5
    for sketches, targets in sample_ds.take(1):
        # save for later
        sample_sketch_batch = sketches
        # sample_target_batch = targets
        
        # save in case we resume training
        np.save(save_dir, sample_sketch_batch)
        
        if use_wandb:
            ## save images of sketches
            fig, axs = plt.subplots(4,4, figsize=(SIZE*4, SIZE*4))
            axs = axs.flatten()
            for img, ax in zip(sketches.numpy(), axs):
                ax.imshow(img, cmap='gray')
                ax.set(xticks=[], yticks=[])
            fig.patch.set_facecolor('white')      
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            
            wandb.log({'Sample Sketch Batch': plt})
            plt.close()

            ## save images of targets
            targets = (targets+1)/2 # [-1,+1] -> [0,1]
            fig, axs = plt.subplots(4,4, figsize=(SIZE*4, SIZE*4))
            axs = axs.flatten()
            for img, ax in zip(targets.numpy(), axs):
                ax.imshow(img)
                ax.set(xticks=[], yticks=[])
            fig.patch.set_facecolor('white')      
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            
            wandb.log({'Sample Target Batch': plt})
            plt.close()
        
    return sample_sketch_batch
    
def show_single_prediction(
    model, input_img, target_img=None, wandb_title=False,
    show_plot=False, patch_cmap='viridis', patch_vmin=-0.5, patch_vmax=1.5):
    """
    Plots model predictions for a single example.
    
    Parameters
    ----------
    model : keras.Model
        Pix2Pix model
    input_img : tensorflow.Tensor
        Sketch to pass into Pix2Pix model, shape=[height, width, 1]
    target_img : tensorflow.Tensor, optional
        Optionally plot the target image alongside.
    wandb_title : str, default False
        Optionally log plot to wandb with this title.
    show_plot : bool, default False
        Draw plot
    patch_cmap : str, matplotlib.colormap, default 'viridis'
        Colormap for PatchGAN output
    patch_vmin : float, default -0.5
        PatchGAN colormap minimum value.
    patch_vmax : float, default 1.5
        PatchGAN colormap maximum value.
    
    Returns
    -------
    
    Examples
    --------
    >>> show_single_prediction(model,
    ...     sample_sketch[tf.newaxis, ...],
    ...     sample_image[tf.newaxis, ...],
    ...     wandb_title=f'Training example',
    ...     patch_cmap=self.patch_cmap)
    
    """
    # model expects [batch size, height, width, channels]
    input_img = input_img[newaxis, ...]
    dis_out, gen_out = model.call(input_img)
    
    if isinstance(target_img, Tensor):
        titles = ['Input Sketch', 'Target Image', 'Predicted Image', 'PatchGAN Output']
        display_list = [input_img, target_img, gen_out, dis_out]
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    else:
        titles = ['Input Sketch', 'Predicted Image', 'PatchGAN Output']
        display_list = [input_img, gen_out, dis_out]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, (ax, title, img) in enumerate(zip(axs, titles, display_list)):
        
        if title in ['Input Sketch', 'Target Image', 'Predicted Image']:
            img = img[0,...]
            img = array_to_img(img)
            ax.imshow(img, cmap='gray')
        elif title == 'PatchGAN Output':
            img = img[0,:,:,0].numpy()
            im = ax.imshow(img, vmin=patch_vmin, vmax=patch_vmax, cmap=patch_cmap)
        
        # axis formatting
        ax.set_title(title)
        ax.axis('off')

    # figure formatting
    cb = fig.colorbar(im, ax=axs[-1], extend='both')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if wandb_title:
        wandb.log({wandb_title: plt})

    if show_plot:
        plt.show()
    plt.close()
    
def show_batch_prediction(model, sketches, wandb_title=False, size=5):
    """Create a grid of predicted outputs for the sample_valid_ds"""
    
    # make predictions
    preds = model.generator(sketches, training=False)
    preds = (preds.numpy() + 1) / 2 # [-1,+1] -> [0,1]
    
    # prepare figure
    fig, axs = plt.subplots(4,4, figsize=(size*4, size*4))
    
    # plot the subplots
    axs = axs.flatten()
    for img, ax in zip(preds, axs):
        ax.imshow(img, cmap='gray')
        ax.set(xticks=[], yticks=[])
        
    # format figure
    fig.patch.set_facecolor('white')      
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # save to wandb
    if wandb_title:
        wandb.log({wandb_title: plt})
    plt.close()
    
def plot_pred_batch(model, dataset, save=False, to_plot=8, size=2, show=True):
    """
    Take to_plot images from dataset.
    Print the input image, the target image and predicted image from model generator.
    Save the image to "save" if desired, and print if show=True.
    size scales the figure
    """

    # make a directory to save the results if necessary
    if save and not os.path.exists(save.parent):
        os.makedirs(save.parent)
        
    # no ticks
    subplot_options = dict(xticks=[], yticks=[])
        
    # take a batch and make predictions
    for x_batch, y_batch in dataset.take(1):
        
        # if one batch has fewer than to_plot then do 1 whole batch
        to_plot = min(to_plot, x_batch.shape[0])
        
        # get predictions and rescale to [0,1]
        preds = (model.generator(x_batch).numpy()+1)/2
        
        x_batch = x_batch.numpy()
        y_batch = (y_batch.numpy()+1)/2
        
        # prepare figure
        fig = plt.Figure(figsize=[size*to_plot, size*3], constrained_layout=True)
        gs = fig.add_gridspec(3,to_plot)
        
        # columns
        for i, imgs in enumerate(zip(x_batch, y_batch, preds)):
            # rows
            for j, img in enumerate(imgs):
                ax = fig.add_subplot(gs[j,i], **subplot_options)
                
                # generator input has 1 channel - b&w sketch
                ax.imshow(img, cmap='gray' if j==0 else 'viridis')
                
                # add titles to left side
                if i==0:
                    ax.set_ylabel(['sketch','target','prediction'][j])
                    
            # stop once plotted enough
            if i == to_plot-1:
                break
        
        # colour between plots
        fig.set_facecolor('white')
        
        if save:
            fig.savefig(str(save))
        if show:
            display(fig)
            
def plot_patches(model, sketch, image, patch_cmap='viridis', patch_vmin=-0.5, patch_vmax=1.5):
    """Plot a figure showing PatchGAN output for a set of generators."""
    
    # Prepare the sub figures
    fig = plt.figure(constrained_layout=True, figsize=[12,8])
    gs0 = fig.add_gridspec(2,1, height_ratios=[1,2], hspace=0)

    gs00 = gs0[0].subgridspec(1,5)
    gs10 = gs0[1].subgridspec(2,5, hspace=0)

    # no ticks
    subplot_options = dict(xticks=[], yticks=[])

    # top 2
    ax1 = fig.add_subplot(gs00[1], title='Sketch', **subplot_options)
    ax2 = fig.add_subplot(gs00[3], title='Image', **subplot_options)

    # Fake generator Ouputs
    ax3 = fig.add_subplot(gs10[0,0], title='Random', **subplot_options)
    ax4 = fig.add_subplot(gs10[0,1], title='Zeros', **subplot_options)
    ax5 = fig.add_subplot(gs10[0,2], title='Identity Generator', **subplot_options)
    ax6 = fig.add_subplot(gs10[0,3], title='Model Generator', **subplot_options)
    ax7 = fig.add_subplot(gs10[0,4], title='Perfect Generator', **subplot_options)
    
    # PatchGan outputs
    ax8  = fig.add_subplot(gs10[1,0], **subplot_options)
    ax9  = fig.add_subplot(gs10[1,1], **subplot_options)
    ax10 = fig.add_subplot(gs10[1,2], **subplot_options)
    ax11 = fig.add_subplot(gs10[1,3], **subplot_options)
    ax12 = fig.add_subplot(gs10[1,4], **subplot_options)
    
    # plot the images
    ax1.imshow(sketch[0].numpy(), cmap='gray')
    ax2.imshow((image[0].numpy()+1)/2)
    
    # create random image - images are [-1,1]
    rand_img = tf.random.uniform(image.shape, -1, 1)
    rand_patch = model.discriminator([sketch, rand_img])
    ax3.imshow((rand_img[0].numpy()+1)/2)
    ax8.imshow(rand_patch[0].numpy(), cmap=patch_cmap, vmin=patch_vmin, vmax=patch_vmax)
    ax8.set_title(rand_patch[0].numpy().mean())
    
    # create image of zeros (img is [-1,1] so "zero" is -1)
    zeros_img = -ones_like(image)
    zeros_patch = model.discriminator([sketch, zeros_img])
    ax4.imshow((zeros_img[0].numpy()+1)/2)
    ax9.imshow(zeros_patch[0].numpy(), cmap=patch_cmap, vmin=patch_vmin, vmax=patch_vmax)
    ax9.set_title(zeros_patch[0].numpy().mean())
    
    # Identity generator
    sketch_rgb = tf.image.grayscale_to_rgb(sketch)*2-1
    ident_patch = model.discriminator([sketch, sketch_rgb])
    ax5.imshow((sketch_rgb[0].numpy()+1)/2)
    ax10.imshow(ident_patch[0].numpy(), cmap=patch_cmap, vmin=patch_vmin, vmax=patch_vmax)
    ax10.set_title(ident_patch[0].numpy().mean())
    
    # model generator
    gen_img = model.generator(sketch)
    gen_img_patch = model.discriminator([sketch, gen_img])
    ax6.imshow((gen_img[0].numpy()+1)/2)
    ax11.imshow(gen_img_patch[0].numpy(), cmap=patch_cmap, vmin=patch_vmin, vmax=patch_vmax)
    ax11.set_title(gen_img_patch[0].numpy().mean())
    
    # Perfect generator
    perfect_patch = model.discriminator([sketch, image])
    ax7.imshow((image[0].numpy()+1)/2)
    im = ax12.imshow(perfect_patch[0].numpy(), cmap=patch_cmap, vmin=patch_vmin, vmax=patch_vmax)
    ax12.set_title(perfect_patch[0].numpy().mean())

    # styling
    cb = fig.colorbar(im, ax=[ax8,ax9,ax10,ax11,ax12], anchor=(0.5,0.5), orientation='horizontal', fraction=.2)
    fig.set_facecolor('white')

    plt.show()
    plt.close()
    
def save_bulk_examples(model, train_ds, valid_ds, viz_path, n_train=10, n_valid=10, **kwargs):
    """Save many examples from train and valid sets to a folder - viz_patch"""
    
    for i in range(n_train):
        plot_pred_batch(model, train_ds, save=viz_path/'train'/f'{i}.png', **kwargs)
        
    for i in range(n_valid):
        plot_pred_batch(model, valid_ds, save=viz_path/'valid'/f'{i}.png', **kwargs)
        
def create_training_gif(image_path, save_dir):
    """Get the files from a wandb directory and create a gif"""
    
    # get list of file paths
    file_names = list(glob.glob(image_path))
    file_names = pd.DataFrame(file_names, columns=['filepaths'])
    # get column of ids
    file_names['number'] = file_names.filepaths.str.extract('Sample Batch\_(\d+)')
    # convert 83->0083 for correct sorting (0083<1111 but 83>1111 sorting alphabetically)
    file_names['number'] = file_names['number'].astype(int).map(lambda x: f"{x:04d}")
    file_names = file_names.sort_values(by='number')
    file_names = file_names.filepaths.to_list()
    
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in file_names]
    img.save(fp=save_dir, format='GIF', append_images=imgs,
            save_all=True, duration=200, loop=0)