# tensorflow-bicubic-downsample
tf.image.resize_images has aliasing when downsampling and does not have gradients for bicubic mode. This implementation fixes those problems.

# Example

tf.images.resize_images | This code | scipy.misc.imresize
--- | --- | ---
![bicubic_mine](https://user-images.githubusercontent.com/12981474/40157448-eff91f06-5953-11e8-9a37-f6b5693fa03f.png) | ![bicubic_tf](https://user-images.githubusercontent.com/12981474/40157450-f247ee22-5953-11e8-9166-9bf979fb4363.png) | ![bicubic_scipy](https://user-images.githubusercontent.com/12981474/40157452-f57d816a-5953-11e8-8e5a-85a591932e3d.png)

# Usage
```
from bicubic_downsample import build_filter, apply_bicubic_downsample

# First, create the bicubic kernel
k = build_filter(factor=4)

# If you want to downsample x which is a tensor with shape [N, H, W, 3]
y = apply_bicubic_downsample(x, filter=k, factor=4)

# y now contains x downsampled to [N, H/4, W/4, 3]
```
