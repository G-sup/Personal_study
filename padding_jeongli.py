from tensorflow.keras.layers import LayerNormalization,Conv2D,Conv2DTranspose,DepthwiseConv2D,LeakyReLU,Input,Layer,InputSpec
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization,GroupNormalization
import tensorflow as tf
import numpy as np
import torch
from torch import nn

# 기본데이터 생성

t = np.array([[[[1, 2, 3], [4, 5, 6],[7,8,9]],[[1, 2, 3], [4, 5, 6],[7,8,9]]]])
print(t)


# ===============================================================================================

# TENSORFLOW  (image 수, Height , Width, channel 수)

inputs = tf.pad(t, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="REFLECT")
# [[image 수 앞장,뒷장],[top_pad, bottom_pad], [left_pad, right_pad],[channel 수 앞채널, 뒷채널]] 아니면 int(전체 pad)
print(t.shape)
print(inputs)

# ===============================================================================================

# PYTORCH  (image 수, channel 수, Height , Width)

t1 = np.transpose(t, (0, 3, 1, 2))
to = nn.ZeroPad2d((0,1,0,1))(torch.from_numpy(t1)) 
# to = nn.ReflectionPad2d((0,1,0,1))(torch.from_numpy(t1)) 
# (padding_left , padding_right , padding_top , padding_bottom )
t2 = np.transpose(to, (0, 2, 3, 1))
print(to)
print(to.shape)
print(t2)

# # ===============================================================================================

# # KERAS  (image 수, Height , Width, channel 수)

y = tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1)))(t)
# (padding=((top_pad, bottom_pad), (left_pad, right_pad))) 아니면 (height_pad, width_pad).
print(y)

class ReflectionPadding2D(Layer):
    def __init__(self, padding=((1, 1),(1, 1)), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        (t_pad ,b_pad),(l_pad,r_pad) = self.padding
        return tf.pad(x, [[0,0], [t_pad,b_pad], [l_pad,r_pad], [0,0] ], 'REFLECT')


y1 = ReflectionPadding2D(padding=((0,1),(0,1)))(t)
print(y1)
