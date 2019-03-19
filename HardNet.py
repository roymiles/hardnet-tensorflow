import tensorflow as tf
import numpy as np
import pickle

class net(object):
    def __init__(self, checkpoint):

        file = open(checkpoint, 'rb')
        pickle_dataset = pickle.load(file)
        file.close()

        self.create_weights(pickle_dataset['weights'])

    def create_weights(self, weights):
        self.weights = {}
        # PyTorch weights stored as [out_channels, in_channels, width, height]
        # We need to reshape that to tensorflow format [filter_height, filter_width, in_channels, out_channels]
        # e.g first layer: torch.Size([32, 1, 3, 3]) -> [3, 3, 1, 32]
        # ... hence the .T transpose

        for name, value in weights.items():
            self.weights[name] = value.T

    def features(self, input):

        features = tf.nn.conv2d(input, self.weights['features.0.weight'], strides=[1,1,1,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.1.running_mean'], self.weights['features.1.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 2

        features = tf.nn.conv2d(features, self.weights['features.3.weight'], strides=[1,1,1,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.4.running_mean'], self.weights['features.4.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 5

        features = tf.nn.conv2d(features, self.weights['features.6.weight'], strides=[1,2,2,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.7.running_mean'], self.weights['features.7.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 8

        features = tf.nn.conv2d(features, self.weights['features.9.weight'], strides=[1,1,1,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.10.running_mean'], self.weights['features.10.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 11

        features = tf.nn.conv2d(features, self.weights['features.12.weight'], strides=[1,2,2,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.13.running_mean'], self.weights['features.13.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 14

        features = tf.nn.conv2d(features, self.weights['features.15.weight'], strides=[1,1,1,1], padding='SAME')
        features = tf.nn.batch_normalization(features, self.weights['features.16.running_mean'], self.weights['features.16.running_var'], None, None, 1e-6)
        features = tf.nn.relu(features) # 17

        # Not needed for now because in test phase
        #features = tf.nn.dropout(features, keep_prob=0.3) # 18
        features = tf.nn.conv2d(features, self.weights['features.19.weight'], strides=[1,1,1,1], padding='VALID')
        features = tf.nn.batch_normalization(features, self.weights['features.20.running_mean'], self.weights['features.20.running_var'], None, None, 1e-6)

        return features

        # PyTorch for reference
        #nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
        #nn.BatchNorm2d(32, affine=False),
        #nn.ReLU(),
        #nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
        #nn.BatchNorm2d(32, affine=False),
        #nn.ReLU(),
        #nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
        #nn.BatchNorm2d(64, affine=False),
        #nn.ReLU(),
        #nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
        #nn.BatchNorm2d(64, affine=False),
        #nn.ReLU(),
        #nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
        #nn.BatchNorm2d(128, affine=False),
        #nn.ReLU(),
        #nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
        #nn.BatchNorm2d(128, affine=False),
        #nn.ReLU(),
        #nn.Dropout(0.3),
        #nn.Conv2d(128, 128, kernel_size=8, bias = False),
        #nn.BatchNorm2d(128, affine=False)

    def input_norm(self, x, eps=1e-6):                      
        x_flatten = tf.layers.flatten(x)
        x_mu, x_std = tf.nn.moments(x_flatten, axes=[1])
        # Add extra dimension
        x_mu = tf.expand_dims(x_mu, axis=1)
        x_std = tf.expand_dims(x_std, axis=1)
        x_norm = (x_flatten - x_mu) / (x_std + eps)         
        return tf.reshape(x_norm, shape=x.shape)                 

    def forward(self, input):

        input_norm = self.input_norm(input)
        # Need to reshape input to: [batch, in_height, in_width, in_channels]
        # Before it was: [batch, in_channels, in_height, in_width] 
        new_shape = (input_norm.shape[0], input_norm.shape[2], input_norm.shape[3], input_norm.shape[1])
        input_norm = tf.reshape(input_norm, shape=new_shape)

        x_features = self.features(input_norm)
        x = tf.reshape(x_features, shape=(x_features.shape[0], -1)) 
        x_norm = tf.math.l2_normalize(x, axis=1)
        return x_norm