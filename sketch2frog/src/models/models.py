from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from src.models.networks import define_generator, define_small_generator, define_discriminator
from src.models.networks import GeneratorLoss, DiscriminatorLoss
from src.models.pix2pix import Pix2Pix

def create_model(
    image_size=[128,128],
    patch_size=8,
    real_label_lower=0.95,
    real_label_upper=1.0,
    model_size="small"):
    
    assert model_size in ["small", "large"]
    
    generator = define_generator(
        sketch_shape=(*image_size,1))

    if model_size=="small":
        discriminator = define_small_generator(
            sketch_shape=(*image_size,1),
            image_shape=(*image_size,3))
        
    elif model_size=="large":
        discriminator = define_generator(
            sketch_shape=(*image_size,1),
            image_shape=(*image_size,3))
    
    discriminator = define_discriminator(
        sketch_shape=(*image_size,1),
        image_shape=(*image_size,3))

    model = Pix2Pix(generator, discriminator,
                    patch_size=patch_size,
                    real_label_lower=real_label_lower,
                    real_label_upper=real_label_upper)
    
    return model

def compile_model(
    model,
    loss="MSE",
    LAMBDA=100,
    g_learning_rate=0.0002,
    g_beta_1=0.5,
    d_learning_rate=0.0002,
    d_beta_1=0.5,
    run_eagerly=None):
    
    # generator loss has tanh activation [-1,1]
    # discriminator loss has no activation
    if loss=="BCE":
        generator_loss_object = BinaryCrossentropy(from_logits=True)
        discriminator_loss_object = BinaryCrossentropy(from_logits=True)
    elif loss=="MSE":
        generator_loss_object = MeanSquaredError()
        discriminator_loss_object = MeanSquaredError()  

    generator_loss = GeneratorLoss(generator_loss_object, LAMBDA)
    discriminator_loss = DiscriminatorLoss(discriminator_loss_object)

    model.compile(
        Adam(lr=g_learning_rate, beta_1=g_beta_1), # g_optimizer
        Adam(lr=d_learning_rate, beta_1=d_beta_1), # d_optimizer
        generator_loss, discriminator_loss, run_eagerly=run_eagerly)

    return model