import pathlib
from src.tools.tools import dotdict

# Add some paths here
PATH = pathlib.Path.cwd()
DATA_DIR = PATH.parent / 'data'
MODELS_DIR = PATH.parent / 'models'
VIZ_DIR = PATH.parent / 'visualisations'

## From main.py
def get_dexined_inference_args():
    """
    Follows code from main.py and updates settings
    for my dataset.
    Returns as a dotdict instead of argparse
    This is needed as argparse can be accessed as arg.model_state
    but a dict cannot. This way we do not need to change any of the DexiNed code.
    
    Using the checkpoints saved in checkpoint_dir, run the model in test mode.
    Use data from test_dir and save results in output_dir with chosen output image size.

    python src/DexiNed-TF2/main.py 
    --model_state="test" --test_dir="data/raw/data-224"--output_dir="data/interim" 
    --checkpoint_dir="src/DexiNed_TF2" --checkpoint="DexiNed23_model.h5" 
    --test_img_height=224 --test_img_width=224 --test_bs=32"""
    
    DATASET_NAME= ['BIPED','MBIPED','BSDS','BSDS300','CID','DCD','MULTICUE',
                    'PASCAL','NYUD','CLASSIC'] # 8

    # 
    TEST_DATA = DATASET_NAME[-1] # MULTICUE=6
    TRAIN_DATA = DATASET_NAME[0]

    # test_data_inf = dataset_info(TEST_DATA, is_linux=in_linux)
    # train_data_inf = dataset_info(TRAIN_DATA, is_linux=in_linux)

    ## When testing we only care about these settings
    # test_data_inf = dataset_info('CLASSIC', is_linux=in_linux)
    # train_data_inf = dataset_info('BIPED', is_linux=in_linux)

    ## whioch returns these dicts
    test_data_inf = {'img_height': 512,
                    'img_width': 512,
                    'test_list': None,
                    'train_list': None,
                    'data_dir': 'data',  # mean_rgb
                    'yita': 0.5}

    train_data_inf = {'img_height': 720,#720
                    'img_width': 1280,#1280
                    'test_list': 'test_rgb.lst',
                    'train_list': 'train_rgb.lst',
                    'data_dir': '../../dataset/BIPED/edges',  # WIN: '../.../dataset/BIPED/edges'
                    'yita': 0.5}


    # Edge detection parameters for feeding the model
    dexined_args = dotdict(
        train_dir=train_data_inf['data_dir'], # path to folder containing images
        test_dir=test_data_inf['data_dir'], # path to folder containing images
        data4train=TRAIN_DATA,
        data4test=TEST_DATA,
        train_list=train_data_inf['train_list'], # SSMIHD: train_rgb_pair.lst, others train_pair.lst
        test_list=test_data_inf['test_list'], # SSMIHD: train_rgb_pair.lst, others train_pair.lst
        model_state='test', # choices=["train", "test", "export"]
        output_dir='results',  # where to put output files
        checkpoint_dir='checkpoints', # directory with checkpoint to resume training from or use for testing
        #
        model_name='DexiNed', # choices=['DexiNed']
        continue_training=False,
        max_epochs=24, # number of training epochs
        summary_freq=100, # update summaries every summary_freq steps
        progress_freq=50, # display progress every progress_freq steps
        display_freq=10, # write current training images every display_freq steps
        scale=None, # scale image before fed DexiNed.0.5, 1.5 
        #
        batch_size=8, # number of images in batch
        test_bs=1, # number of images in test batch
        batch_normalization=True, # use batch norm
        image_height=400, # scale images to this size before cropping to 256x256
        image_width=400, # scale images to this size before cropping to 256x256
        crop_img=False, # 4Training: True crop image, False resize image
        test_img_height=test_data_inf["img_height"], # network input height size
        test_img_width=test_data_inf["img_width"], # network input height size
        #
        lr=0.0001, # learning rate for adam 1e-4
        beta1=0.5, # momentum term of adam
        l1_weight=100.0, # weight on L1 term for generator gradient
        gan_weight=1.0, # weight on GAN term for generator gradient
        rgbn_mean=[103.939,116.779,123.68, 137.86], # pixels mean
        checkpoint='DexiNed19_model.h5', # checkpoint Name
    )

    # overwrite with our settings
    # if running main.py from the sketch2frog directory
    dexined_args.update(dotdict(
        model_state="test",
        test_dir="data/raw/data-224",
        output_dir="data/interim",
        checkpoint_dir="src/DexiNed_TF2",
        checkpoint="DexiNed23_model.h5",
        test_img_height=224,
        test_img_width=224,
        test_bs=32
    ))
    
    return dexined_args