import tensorflow as tf
from utils import conv2d, relu, ortho_init
import numpy as np

def build_cnn(states, reuse=False,init=None):
    """
    Builds ACER network, returns the shared layer
    """
    #init = tf.initializers.random_normal(0,0.1)

    with tf.variable_scope("shared_policy_net"):
        with tf.variable_scope("conv_layer_1"):
            conv_l1 = relu(conv2d(states, 8, 4, 32,init))
        with tf.variable_scope("conv_layer_2"):
            conv_l2 = relu(conv2d(conv_l1, 4, 2, 64,init))
        with tf.variable_scope("conv_layer_3"):
            conv_l3 = relu(conv2d(conv_l2, 3, 1, 64,init))
        with tf.name_scope("flatten"):
            conv_l3_flat = tf.layers.flatten(conv_l3)
        with tf.variable_scope("shared_fully_connected"):
            shared_fc = tf.layers.dense(
                inputs=conv_l3_flat,
                units=512,
                activation=tf.nn.relu,
                kernel_initializer = init,
            )
    return shared_fc




# CNN Proposed by Mnih for A3C atari
def build_mnih_A3C (state):
    # First convolutional layer
    with tf.variable_scope('conv_l_1'):
        conv1 = tf.contrib.layers.convolution2d(inputs=state,
                                                num_outputs=16, kernel_size=[8, 8], stride=[4, 4], padding="VALID",
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.zeros_initializer())

    # Second convolutional layer
    with tf.variable_scope('conv2_l_2'):
        conv2 = tf.contrib.layers.convolution2d(inputs=conv1, num_outputs=32,
                                                kernel_size=[4, 4], stride=[2, 2], padding="VALID",
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.zeros_initializer())

    # Flatten the network
    with tf.name_scope('conv_flatten'):
        flatten = tf.contrib.layers.flatten(inputs=conv2)

    # Fully connected layer with 256 hidden units
    with tf.variable_scope('fully_connected_shared'):
        fc1 = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=256,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.zeros_initializer())

    return fc1

# split actor critic with only FCs for cartpole'ish problems.
def build_simple_fc(states):
    with tf.variable_scope('actor_l1'):
        fc_actor = tf.layers.dense(
            inputs=states,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=ortho_init(np.sqrt(2)),
        )
    with tf.variable_scope('actor_l2'):
        fc_actor2 = tf.layers.dense(
            inputs=fc_actor,
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=ortho_init(np.sqrt(2)),

        )
    with tf.variable_scope('critic_l1'):
        fc_critic = tf.layers.dense(
            inputs=states,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=ortho_init(np.sqrt(2)),
        )
    with tf.variable_scope('critic_l2'):
        fc_critic2 = tf.layers.dense(
            inputs=fc_critic,
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=ortho_init(np.sqrt(2)),
        )

    return fc_actor2, fc_critic2

def build_shared_fc(states):
    with tf.variable_scope('actor_l1'):
        fc_actor = tf.layers.dense(
            inputs=states,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=ortho_init(np.sqrt(2)), 
        )
    with tf.variable_scope('actor_l2'):
        fc_actor2 = tf.layers.dense(
            inputs=fc_actor,
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=ortho_init(np.sqrt(2)),
        )
    return fc_actor2


def create_network2(states, ac_space, network_type, v_space):

    if network_type == "cnn":
        l_shared = build_cnn(states)
        with tf.variable_scope("Logits"):
            scores = tf.layers.dense(
                inputs=l_shared,
                units=ac_space,
                kernel_initializer=ortho_init(0.01),
            )
        with tf.variable_scope("V_fc"):
            quality = tf.layers.dense(
                inputs=l_shared,
                units=v_space,
                kernel_initializer=ortho_init(),
            )
        pi = tf.nn.softmax(scores)

        return pi, quality

    if network_type == "ff":
        l_ac, l_ct = build_simple_fc(states)
        with tf.variable_scope("Logits"):
            scores = tf.layers.dense(
                inputs=l_ac,
                units=ac_space,
                kernel_initializer=ortho_init(0.01),
            )
        with tf.variable_scope("V_fc"):
            quality = tf.layers.dense(
                inputs=l_ct,
                units=v_space,
                kernel_initializer=ortho_init(),
            )
        pi = tf.nn.softmax(scores)

        return pi, quality
    if network_type == "mnih":
        l_shared = build_mnih_A3C(states)
        with tf.variable_scope("Logits"):
            scores = tf.layers.dense(
                inputs=l_shared,
                units=ac_space,
                kernel_initializer=ortho_init(0.01),
            )
        with tf.variable_scope("V_fc"):
            quality = tf.layers.dense(
                inputs=l_shared,
                units=v_space,
                kernel_initializer=ortho_init(),
            )
        pi = tf.nn.softmax(scores)

        return pi, quality

    
def create_network(states, ac_space, network_type, v_space):

    if network_type == "cnn":
        l_shared = build_cnn(states)
        with tf.variable_scope("Logits"):
            scores = tf.layers.dense(
                inputs=l_shared,
                units=ac_space,
            )
        with tf.variable_scope("V_fc"):
            quality = tf.layers.dense(
                inputs=l_shared,
                units=v_space,
            )
        pi = tf.nn.softmax(scores)

        return pi, quality

    if network_type == "ff":
        l_ac, l_ct = build_simple_fc(states)
        with tf.variable_scope("Logits"):
            scores = tf.layers.dense(
                inputs=l_ac,
                units=ac_space,
                kernel_initializer=ortho_init(0.01),
            )
        with tf.variable_scope("V_fc"):
            quality = tf.layers.dense(
                inputs=l_ct,
                units=v_space,
                kernel_initializer=ortho_init(),
            )
        pi = tf.nn.softmax(scores)

        return pi, quality
    if network_type == "mnih":
        l_shared = build_mnih_A3C(states)
        with tf.variable_scope("Logits"):
            scores = tf.layers.dense(
                inputs=l_shared,
                units=ac_space,
                kernel_initializer=ortho_init(0.01),
            )
        with tf.variable_scope("V_fc"):
            quality = tf.layers.dense(
                inputs=l_shared,
                units=v_space,
                kernel_initializer=ortho_init(),
            )
        pi = tf.nn.softmax(scores)

        return pi, quality
