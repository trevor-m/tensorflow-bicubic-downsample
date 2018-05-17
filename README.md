# tensorflow-bicubic-downsample
tf.image.resize_images has aliasing when downsampling and does not have gradients for bicubic mode. This implementation fixes those problems.

# Usage
```
from bicubic_downsample import build_filter, apply_bicubic_downsample

# First, create the bicubic kernel
k = build_filter(factor=4)

# If you want to downsample x which is a tensor with shape [N, H, W, 3]
y = apply_bicubic_downsample(x, filter=k, factor=4)

# y now contains x downsampled to [N, H/4, W/4, 3]
```
