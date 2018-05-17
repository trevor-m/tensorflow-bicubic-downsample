# tensorflow-bicubic-downsample
tf.image.resize_images has aliasing when downsampling and does not have gradients for bicubic mode. This implementation fixes those problems.
