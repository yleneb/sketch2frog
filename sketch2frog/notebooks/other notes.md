Useful links:
https://ml4a.github.io/guides/Pix2Pix/
https://heartbeat.fritz.ai/training-a-tiny-pix2pix-gan-for-snapchat-c6062e86204c
https://arxiv.org/pdf/1504.06375.pdf - Holistically-Nested Edge Detection
https://affinelayer.com/pixsrv/ - pix2pix demo
https://github.com/jonshamir/frog-dataset


pix2pix demo with implementation code
https://zaidalyafeai.github.io/pix2pix/cats.html#
https://github.com/zaidalyafeai/zaidalyafeai.github.io/tree/master/pix2pix

GauGAN - SPADE
https://arxiv.org/pdf/1903.07291.pdf

CycleGAN manga colorization
https://github.com/OValery16/Manga-colorization---cycle-gan 

pix2pix + SPADE

# Slow Down / Memory Leak

On increasing the input image size (128->256) and subsequently reducing batch size (64->16) I noticed that after ~10 epochs training would slow down significantly along with GPU utilisation. The slow down was sufficient enough for full training to become unfeasibly long. This was accompanied by a rapid increase in memory use by a python process, "GPU 0 - Copy", responsible for moving data between the system and the GPU.

Through experimenting I know that this is not due to:
- my model (I swapped to a simple CNN)
- part of training - tried just making predictions
- the data augmentation (tried without)
- cache().shuffle().repeat().batch().prefetch()

So the problem must be either due to tf.data or part of the code below. However, these steps are followed by .cache() so should only be performed once. Setting run_eagerly=True in model.compile() solves this problem, but training is 75% slower. Alternatively, I can train for a bit, save weights, restart kernel, and continue from the saved weights.

```python
def decode_sketch(sketch_path, image_shape):
    # takes a png file path and returns a tensor
    sketch = tf.io.read_file(sketch_path)
    sketch = tf.io.decode_png(sketch)
    sketch = tf.cast(sketch, tf.float32) / 255
    return tf.image.resize(sketch, image_shape, method='nearest')

def decode_image(image_path, image_shape):
    # takes a png file path and returns a tensor
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=3) / 255 # [0,255] -> [0,1]
    img = img*2 - 1 # [0,1] -> [-1,1]  (range of tanh)
    img = tf.cast(img, tf.float32)
    return tf.image.resize(img, image_shape, method='nearest')

def process_path(fname, image_shape=[256,256]):
    # takes the sketch image path and returns tensors for the sketch and source image
    sketch_path = tf.strings.join([str(SKETCH_DIR), '\\', fname])
    image_path = tf.strings.join([str(IMAGES_DIR), '\\', fname])
    sketch = decode_sketch(sketch_path, image_shape)
    image = decode_image(image_path, image_shape)
    return sketch, image
    
list_ds = tf.data.Dataset.from_tensor_slices(list_ds)
list_ds = list_ds.shuffle(IMAGE_COUNT, seed=SEED, reshuffle_each_iteration=False)
train_ds = list_ds.skip(VALID_LENGTH)
train_ds = train_ds.map(functools.partial(process_path, image_shape=IMAGE_SIZE), num_parallel_calls=AUTOTUNE)```