import tensorflow as tf
import numpy as np
import HardNet as HardNet

pickle_checkpoint = "C:\\Users\\Roy\\source\\repos\\roymiles\\bench-match\\hardnet_checkpoint.pickle"

h = HardNet.net(pickle_checkpoint)

# Test it
# torch.Size([500, 1, 32, 32])
input = tf.constant(np.random.rand(500, 1, 32, 32))

with tf.Session() as sess:
    out = sess.run(h.forward(input))
    print("Out: {}".format(out))
    print("The end")
