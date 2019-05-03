# tensorflow-bicubic-downsample
tf.image.resize_images has aliasing when downsampling and does not define gradients for bicubic mode. This implementation fixes those problems.

# Example
These images have been downsampled by a factor of 4 from the original. The results from this code matches the scipy.misc.imresize results exactly.

Method | Result | Comments
--- | --- | ---
Original | <img src="https://user-images.githubusercontent.com/12981474/40157591-b5260abe-5954-11e8-8218-25ee937425ec.png" width="200"> | This is the original full res image.
tf.images.resize_images | <img src="https://user-images.githubusercontent.com/12981474/40157450-f247ee22-5953-11e8-9166-9bf979fb4363.png" width="200"> | TF's implementation has aliasing
scipy.misc.imresize | <img src="https://user-images.githubusercontent.com/12981474/40157452-f57d816a-5953-11e8-8e5a-85a591932e3d.png" width="200"> | Proper bicubic downsampling
This code | <img src="https://user-images.githubusercontent.com/12981474/40157448-eff91f06-5953-11e8-9a37-f6b5693fa03f.png" width="200"> | Matches scipy exactly

# Usage
```python
from bicubic_downsample import build_filter, apply_bicubic_downsample

# First, create the bicubic kernel. This can be reused in multiple downsample operations
k = build_filter(factor=4)

# Downsample x which is a tensor with shape [N, H, W, 3]
y = apply_bicubic_downsample(x, filter=k, factor=4)

# y now contains x downsampled to [N, H/4, W/4, 3]
```
