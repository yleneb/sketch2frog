import tensorflow as tf
from tensorflow.keras import layers

######################################################
# Helper Functions
######################################################

# The training=True is intentional here since you want the batch statistics,
# while running the model on the test dataset.
# If you use training=False, you get the accumulated statistics
# learned from the training dataset (which you don't want).

# UNET (generator)
def encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = tf.initializers.RandomNormal(stddev=0.02)

    # add downsampling layer
    # if using batchnorm, then no bias needed
    g = layers.Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init, use_bias=not(batchnorm))(layer_in)

    # optional batchnorm
    if batchnorm:
        # always operate in training mode, even when used during inference
        g = layers.BatchNormalization()(g, training=True)

    # All ReLUs in the encoder are leaky, with slope 0.2
    g = layers.LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=False):
    # weight initialization
    init = tf.initializers.RandomNormal(stddev=0.02)

    # add upsampling layer
    g = layers.Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init, use_bias=False)(layer_in)

    # add batch norm always
    g = layers.BatchNormalization()(g, training=True)
    # optional dropout
    if dropout:
        g = layers.Dropout(dropout)(g, training=True)
    # merge with skip connection
    g = layers.Concatenate()([g, skip_in])
    # ReLUs in the decoder are not leaky
    g = layers.ReLU()(g)
    return g

# define_G from name (128 vs 256)

######################################################
# Classes
######################################################

class GeneratorLoss(tf.keras.losses.Loss):
    """
    loss=['binary_crossentropy', 'mae']
    loss_weights=[1, 100]
    The Pix2Pix generator uses a composite of Adversarial Loss (BCE) and L1 Loss (Pixel Distance Loss)

    The adversarial loss influences whether the generator
    model can output images that are plausible in the target domain.
    This is calculated from the output of the discriminator.
    Non-saturating GAN Loss:
        maximize the probability of images being predicted as real.
    Hence y_target is an array of ones.
    
    The L1 loss regularizes the generator model to output images
    that are a plausible translation of the source image.
    This is calculated from the output of the generator.
    """   
    def __init__(self, loss_object, LAMBDA=100):
        super(GeneratorLoss, self).__init__()
        self.loss_object = loss_object
        self.LAMBDA = LAMBDA
        
    def call(self, y_pred, y_target, out_img, target_img):
        # binary cross entropy - patches
        gan_loss = self.loss_object(y_target, y_pred)

        # Pixel distance loss (L1)
        l1_loss = tf.reduce_mean(tf.abs(target_img - out_img))

        # total loss is weighted through lambda
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss
    

class DiscriminatorLoss(tf.keras.losses.Loss):
    """Combine the discriminator losses from both the real and fake images"""
    def __init__(self, loss_object):
        super(DiscriminatorLoss, self).__init__()
        self.loss_object = loss_object
    
    def call(self, d_real_pred, d_real_target, d_fake_pred, d_fake_target):
        real_loss = self.loss_object(d_real_target, d_real_pred)
        fake_loss = self.loss_object(d_fake_target, d_fake_pred)
        
        return real_loss + fake_loss

######################################################
# Encoder and Decoders
######################################################

def define_generator(sketch_shape=(128,128,1), dropout_pct=0.5):
    """Modified structure for 128x128 input, simply removed one layer from each side"""
    # weight initialization
    init = tf.initializers.RandomNormal(stddev=0.02)
    # image input
    in_image = layers.Input(shape=sketch_shape) # (batch_size, 128, 128, 1)

    # encoder model: C64-C128-C256-C512-C512-C512-C512
    # BatchNorm is not applied to the first C64 layer in the encoder
    e1 = encoder_block(in_image, 64, batchnorm=False) # (batch_size, 64, 64, 64)
    e2 = encoder_block(e1, 128)  # (batch_size, 32, 32, 128)
    e3 = encoder_block(e2, 256)  # (batch_size, 16, 16, 256)
    e4 = encoder_block(e3, 512)  # (batch_size,  8,  8, 512)
    e5 = encoder_block(e4, 512)  # (batch_size,  4,  4, 512)
    e6 = encoder_block(e5, 512)  # (batch_size,  2,  2, 512)

    # bottleneck, no batch norm, relu not leaky
    b = layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
    b = layers.ReLU()(b) # (batch_size, 1, 1, 512)

    # decoder model: CD512-CD1024-CD1024-C1024-C512-C256-C128
    # N.B. only first 3 use dropout
    d1 = decoder_block(b,  e6, 512, dropout_pct)    # (batch_size,   2,   2, 1024) # doubles filters due to concat
    d2 = decoder_block(d1, e5, 512, dropout_pct)    # (batch_size,   4,   4, 1024)
    d3 = decoder_block(d2, e4, 512, dropout_pct)    # (batch_size,   8,   8, 1024)
    d4 = decoder_block(d3, e3, 256, dropout=False)  # (batch_size,  16,  16,  512)
    d5 = decoder_block(d4, e2, 128, dropout=False)  # (batch_size,  32,  32,  256)
    d6 = decoder_block(d5, e1,  64, dropout=False)  # (batch_size,  64,  64,  128)

    # output (batch_size, 128, 128, 3)
    g = layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d6)
    out_image = layers.Activation('tanh')(g)

    return tf.keras.Model(in_image, out_image)

# PatchGAN (discriminator)
def define_discriminator(sketch_shape=(128,128,1), image_shape=(128,128,3)):
    # weight initialization
    init = tf.initializers.RandomNormal(stddev=0.02)
    # source image input - same as input to generator
    in_src_image = layers.Input(shape=sketch_shape) # (batch_size, 128, 128, 1)
    # target image input - output of generator or a real image
    in_target_image = layers.Input(shape=image_shape) # (batch_size, 128, 128, 3)
    # concatenate images channel-wise (batch_size, 128, 128, 1+3=4)
    merged = layers.Concatenate()([in_src_image, in_target_image])

    # Ck denote a Convolution-BatchNorm-ReLU layer with k filters
    # model: C64-C128-C256-C512
    # C64 - BatchNorm is not applied to the first C64 layer (batch_size, 64, 64, 64)
    d = layers.Conv2D(64,  (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = layers.LeakyReLU(alpha=0.2)(d) # All ReLUs are leaky, with slope 0.2
    # C128 (batch_size, 32, 32, 128)
    d = layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # C256 (batch_size, 16, 16, 256)
    d = layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # C512 (batch_size, 8, 8, 512)
    d = layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # second last output layer - NB no strides (batch_size, 8, 8, 512)
    d = layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # patch output (batch_size, 8, 8, 1)
    patch_out = layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    # no activation if we want to use label smoothing
    # patch_out = layers.Activation('sigmoid')(d) # output [0,1]

    # define model 
    return tf.keras.Model([in_src_image, in_target_image], patch_out)