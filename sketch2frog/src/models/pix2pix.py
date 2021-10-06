import tensorflow as tf

class Pix2Pix(tf.keras.Model):
    """
    Inspired by this twitter post from the creator of keras
    https://twitter.com/fchollet/status/1250622989541838848/photo/1
    and the pix2pix tensorflow tutorial
    https://www.tensorflow.org/tutorials/generative/pix2pix

    patch_size is the size of the output of the PatchGAN discriminator
    """
    def __init__(self, generator, discriminator, patch_size=8, real_label_lower=0.8, real_label_upper=1.2, name="pix2pix", **kwargs):
        super(Pix2Pix, self).__init__(name=name, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.patch_size = patch_size
        self.real_label_lower = real_label_lower
        self.real_label_upper = real_label_upper
        
    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn, run_eagerly=None):
        super(Pix2Pix, self).compile()
        self.g_optimizer = g_optimizer # Adam(lr=0.0002, beta_1=0.5)
        self.d_optimizer = d_optimizer # Adam(lr=0.0002, beta_1=0.5)
        self.g_loss_fn = g_loss_fn # loss=['binary_crossentropy', 'mae'], loss_weights=[1, 100]
        self.d_loss_fn = d_loss_fn # bce, loss_weights=[0.5]
        self._run_eagerly = run_eagerly

    def call(self, data, training=False):
        gen_out = self.generator(data, training=training)
        dis_out = self.discriminator([data, gen_out], training=training)
        return dis_out, gen_out

    def train_step(self, data):
        """Chollet's approach trains the discriminator, then generates new images and trains the generator
        Each time overwriting the GradientTape and predictions.
        
        Meanwhile this approach (pix2pix tf tutorial) creates a gen_tape and disc_tape simultaneously.
        Thus the gradients for training the gen and disc must both be available at the same time.
        I suspect that this approach will require more VRAM,
        but since it only has a single forward pass through the generator it may save time overall.
        
        One sided Label Smoothing would smooth the positive target only.
        Only when training the Discriminator.
        When training the Generator we do not smooth the labels.
        This is because label smoothing is a form of regularisation to slow down the discriminator.
        """
        
        # unpack data into sketches (input) and images (target)
        input_batch, target_batch = data
        batch_size = tf.shape(input_batch)[0]
        
        # create patchgan labels
        labels_real = tf.ones((batch_size, self.patch_size, self.patch_size, 1))
        labels_fake = tf.zeros((batch_size, self.patch_size, self.patch_size, 1))
        
        # create smoothed patchgan labels for reals
        labels_real_smoothed = tf.random.uniform(
            shape=(batch_size, self.patch_size, self.patch_size, 1),
            minval=self.real_label_lower,
            maxval=self.real_label_upper)
        
        # complete the forward pass and calculate the losses
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generate fake images once
            gen_output = self.generator(input_batch, training=True)
            
            # predict on reals and fakes separately due to Batchnorm layers
            disc_real_pred = self.discriminator([input_batch, target_batch], training=True)
            disc_fake_pred = self.discriminator([input_batch, gen_output], training=True)
            
            # y_target is an array of ones to maximise the probability of generated images
            # being predicted as real - non-saturating GAN loss
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.g_loss_fn(
                y_pred=disc_fake_pred, y_target=labels_real,
                out_img=gen_output,  target_img=target_batch)
            
            # gen_total_loss, gen_gan_loss, gen_l1_loss = self.g_loss_fn(
            #     y_true=[labels_real, target_batch],
            #     y_pred=[disc_fake_pred, gen_output])
            
            # calculate the discriminator loss, combining reals and fakes
            # with one-sided label smoothing on the reals.
            disc_loss = self.d_loss_fn(
                d_real_pred=disc_real_pred, d_real_target=labels_real_smoothed,
                d_fake_pred=disc_fake_pred, d_fake_target=labels_fake)
        
        # calculate gradients and update weights
        gen_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_weights)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_weights))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_weights))
        
        return {
            'd_loss': disc_loss,
            'gen_total_loss': gen_total_loss,
            'gen_gan_loss': gen_gan_loss,
            'gen_l1_loss': gen_l1_loss}
        
    def test_step(self, data):
        """Customize the model.evaluate() function to get the validation losses"""
        
        # unpack data into sketches (input) and images (target)
        input_batch, target_batch = data
        batch_size = tf.shape(input_batch)[0]
        
        # create patchgan labels - don't used smoothing for evaluation
        labels_real = tf.ones((batch_size, self.patch_size, self.patch_size, 1))
        labels_fake = tf.zeros((batch_size, self.patch_size, self.patch_size, 1))
        
        # complete the forward pass and calculate the losses
        # generate fake images once
        gen_output = self.generator(input_batch)
        
        # predict on reals and fakes separately due to Batchnorm layers
        disc_real_pred = self.discriminator([input_batch, target_batch])
        disc_fake_pred = self.discriminator([input_batch, gen_output])
        
        # y_target is an array of ones to maximise the probability of generated images
        # being predicted as real - non-saturating GAN loss
        gen_total_loss, gen_gan_loss, gen_l1_loss = self.g_loss_fn(
            y_pred=disc_fake_pred, y_target=labels_real,
            out_img=gen_output,  target_img=target_batch)
        
        # calculate the discriminator loss, combining reals and fakes
        # with one-sided label smoothing on the reals.
        disc_loss = self.d_loss_fn(
            d_real_pred=disc_real_pred, d_real_target=labels_real,
            d_fake_pred=disc_fake_pred, d_fake_target=labels_fake)
        
        return {
            'd_loss': disc_loss,
            'gen_total_loss': gen_total_loss,
            'gen_gan_loss': gen_gan_loss,
            'gen_l1_loss': gen_l1_loss}