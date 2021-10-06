import tensorflow as tf
import albumentations as A
from cv2 import BORDER_CONSTANT
import functools
import os

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

def _process_path(fname, image_shape, sketch_dir, images_dir):
    # takes the sketch image path and returns tensors for the sketch and source image
    sketch_path = tf.strings.join([str(sketch_dir), '\\', fname])
    image_path = tf.strings.join([str(images_dir), '\\', fname])
    sketch = decode_sketch(sketch_path, image_shape)
    image = decode_image(image_path, image_shape)
    return sketch, image

def _aug_fn(sketch, image, transforms):
    # both images should be transformed the same
    data = {"image":sketch, 'image1':image}
    aug_data = transforms(**data)
    aug_sketch = aug_data["image"]
    aug_image = aug_data["image1"]
    return aug_sketch, aug_image

def data_augmentation_wrapper(transforms):
    # returns the data_augmentation function for our given transforms
    aug_fn = functools.partial(_aug_fn, transforms=transforms)
    def data_augmentation(sketch, image):
        # wrapper around augmentation function for tensorflow compatibility
        aug_sketch, aug_image = tf.numpy_function(
            func=aug_fn, inp=[sketch, image], Tout=[tf.float32, tf.float32])
        return aug_sketch, aug_image
    return data_augmentation

# restore dataset shapes
def set_shapes(sketch, image, image_shape=[128,128]):
    """The datasets loses its shape after applying a tf.numpy_function"""
    sketch.set_shape([*image_shape, 1])
    image.set_shape([*image_shape, 3])
    return sketch, image

def get_datasets(image_size, sketch_dir, images_dir, train_length, valid_length,
                 seed, is_test, shuffle_buffer_size,
                 batch_size, prefetch_buffer_size, num_parallel_calls):
    
    # Instantiate augmentations - jitter, rotate, mirror
    transforms = A.Compose([
        A.PadIfNeeded(min_height=image_size[0]+20, min_width=image_size[1]+20, border_mode=BORDER_CONSTANT, value=[1,1,1]),
        A.Rotate(limit=40, border_mode=BORDER_CONSTANT, value=[1,1,1]),
        A.RandomCrop(*image_size),
        A.HorizontalFlip(),
        #A.Affine(shear=[-45,45], mode=cv2.BORDER_CONSTANT, cval=[1,1,1])
        ], additional_targets={'image1':'image'})
    
    # get a list of all the sketch paths
    list_ds = os.listdir(sketch_dir)

    # use a subset for testing
    if is_test:
        list_ds = list_ds[:train_length+valid_length]

    # create tensorflow dataset
    list_ds = tf.data.Dataset.from_tensor_slices(list_ds)
    list_ds = list_ds.shuffle(train_length+valid_length, seed=seed, reshuffle_each_iteration=False)

    # train / valid split
    train_ds = list_ds.skip(valid_length)
    valid_ds = list_ds.take(valid_length)

    # process filepaths to images
    process_path = functools.partial(_process_path, image_shape=image_size, sketch_dir=sketch_dir, images_dir=images_dir)
    train_ds = train_ds.map(process_path, num_parallel_calls=num_parallel_calls)
    valid_ds = valid_ds.map(process_path, num_parallel_calls=num_parallel_calls)

    # cache datasets before performing data augmentation
    train_ds = (train_ds
                .cache()      
                .shuffle(shuffle_buffer_size)
                .repeat()
                .map(data_augmentation_wrapper(transforms), num_parallel_calls=num_parallel_calls)
                .map(functools.partial(set_shapes, image_shape=image_size), num_parallel_calls=num_parallel_calls)
                .batch(batch_size)
                .prefetch(buffer_size=prefetch_buffer_size))

    valid_ds = (valid_ds
                .cache()
                .repeat()
                .batch(batch_size))
    
    return train_ds, valid_ds