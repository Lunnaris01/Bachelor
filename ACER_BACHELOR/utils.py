import numpy as np
import tensorflow as tf
from skimage import transform, color


def conv2d(x, kernel, stride, filters, init=None):
    return tf.layers.conv2d(inputs=x,
                            strides=stride,
                            filters=filters,
                            kernel_size=kernel,
                            padding='VALID',
                            kernel_initializer=init,
                            )



def relu(x):
    return tf.nn.relu(x)


def to_grayscale(img):
    return color.rgb2grey(img) * 255


def downsample(img):
    return transform.resize(img[34::], (84, 84));


# Atari games:
# 0 - Breakout

# def preprocess(states,game=0):
#    if game == 0:
#        p_state=[to_grayscale(downsample(x)).astype(np.uint8) for x in states]
#        return np.rollaxis(np.array(p_state),0,3)

def preprocess(state, game=0):
    if game == 0:
        return to_grayscale(downsample(state)).astype(np.uint8)


def discount_rewards(rewards, discount=0.97):
    discounted_r = np.zeros(len(rewards))
    discounted_r[-1] = rewards[-1]
    for it in range(len(rewards) - 1)[::-1]:
        discounted_r[it] = rewards[it] + discount * discounted_r[it + 1]
    return discounted_r


# currently unused, needs fixing.
def step_env(env, action):
    rawstate = []
    done = False
    reward = 0
    info = None
    for i in range(4):
        s, r, done, info = env.step(action)
        rawstate.append(s)
        reward += r
        if done and len(rawstate) < 4:
            for _ in range(len(rawstate) - 4):
                rawstate.append(rawstate[len(rawstate) - 1])
            break
    return (preprocess(rawstate), np.sign(reward), done, info)


# Orthogonal Init from openAI baselines, which is based on the Lasagne implementation.

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def q_retrace(rewards, done, qualities, values, importance_weights, DISCOUNT):
    importance_weights_t = np.minimum(1.0, importance_weights)
    next_ret = 0
    if not done:
        next_ret = values[-1]
    retrace_targets = []
    for i in range(len(rewards, ) - 1, -1, -1):
        next_ret = rewards[i] + DISCOUNT * next_ret
        retrace_targets.append(next_ret)
        next_ret = (importance_weights_t[i] * (next_ret - qualities[i])) + values[i]
    retrace_targets = retrace_targets[::-1]
    return retrace_targets


def lambda_return(rewards, values, done, DISCOUNT, lambda_w):
    targets = []
    target = 0
    if not done:
        target = values[-1]
    for it in range(len(rewards) - 1, -1, -1):
        targets.append(rewards[it] + lambda_w * DISCOUNT * target + ((1 - lambda_w) * DISCOUNT * values[it + 1]))
        target = targets[-1]
    targets = targets[::-1]
    targets = np.reshape(targets, [-1])

    return targets

def rollout(agent, env, states, done, RETURN_STEPS):
    if done:
        states = [env.reset()]
    done = False
    actions, rewards, mus = [], [], []
    while not done and len(states) < RETURN_STEPS:
        action, pi = agent.get_action_with_probs(states[-1])
        state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        mus.append(pi)
    return (np.array(states), actions, rewards, mus, done)

