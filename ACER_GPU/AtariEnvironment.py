from utils import preprocess
import numpy as np

class Atari_Environment(object):
    """
    
    Atari Environment wrapper, takes in an environment created by e.g. openai gym.
    Frameskipping needs to be provided by the env.
    Stacks n frames.
    
    Frames are saved to the framebuffer after preprocessing as int8 greyscaled pictures (see thesis)
    
    """
    def __init__(self, env, use_every_n_frame = 1, stackframes = 4, clipreward = False):
        """
        Initialize the environment, taking in some kind of Atari Environment like the OpenAI Gym environment.
        The env needs to provice a step and reset function, have an action_space and observation_space variable.
        Clips reward to the intervall [-1,1] if clipreward is True.
        """
        self.env = env
        self.frameskip = use_every_n_frame
        self.steps_per_state = use_every_n_frame*stackframes
        self.stackframes = stackframes
        self.clipreward = clipreward
        self.action_space=env.action_space
        self.observation_space = env.observation_space
        self.observation_space.shape = (84,84,4)
        self.framebuffer = []

    def reset(self):
        """
        Reset the Environment, returns the inital state.
        """
        state = self.env.reset()
        state = preprocess(state)
        state = np.stack([state] * self.stackframes, axis=2)
        self.framebuffer = state
        return self.framebuffer
    
    def step(self,action):
        """
        Takes in an action and gives back the new state, the reward and the terminal signal
        """
        state, reward, done, info = self.env.step(action)
        state = preprocess(state)
		#This disposes of the first element, and adds the new state to the right dimension.
        self.framebuffer = np.append(self.framebuffer[:,:,1:], np.expand_dims(state, 2), axis=2)
        if self.clipreward:
            reward=np.sign(reward)
        return self.framebuffer,reward,done,info 

    def render(self):
        self.env.render()
