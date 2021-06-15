from tensorflow.keras.layers import LayerNormalization,Conv2D,Conv2DTranspose,DepthwiseConv2D,LeakyReLU,Input
from tensorflow.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization,GroupNormalization
import tensorflow as tf
import numpy as np
import torch
from torch import nn

# 기본데이터 생성
t = np.array([[[[1, 1, 1], [2, 2, 2]],[[1, 1, 1], [2, 2, 2]]]])
print(t)
# ===============================================================================================
# TENSORFLOW  (image 수, Height , Width, channel 수)
inputs = tf.pad(t, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="CONSTANT")
# [[image 수 앞장,뒷장],[top_pad, bottom_pad], [left_pad, right_pad],[channel 수 앞채널, 뒷채널]] 아니면 int(전체 pad)
print(t.shape)
print(inputs)

# ===============================================================================================
# PYTORCH  (image 수, channel 수, Height , Width)
t1 = np.transpose(t, (0, 3, 1, 2))
to = nn.ZeroPad2d((0,1,0,1))(torch.from_numpy(t1)) 
# (padding_left , padding_right , padding_top , padding_bottom )
t2 = np.transpose(to, (0, 2, 3, 1))
print(to)
print(to.shape)
print(t2)

# ===============================================================================================
# KERAS  (image 수, Height , Width, channel 수)
y = tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1)))(t)
# (padding=((top_pad, bottom_pad), (left_pad, right_pad))) 아니면 (height_pad, width_pad).
print(y)

