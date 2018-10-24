from utils import preprocess
import numpy as np

class AA_Atari_Environment(object):
    #
    def __init__(self, env, use_every_n_frame = 1, stackframes = 4, clipreward = False, pixelwhitening=0, pixelwhiteningw=0,
                noise=0):
        
        self.env = env
        self.frameskip = use_every_n_frame
        self.steps_per_state = use_every_n_frame*stackframes
        self.stackframes = stackframes
        self.clipreward = clipreward
        self.action_space=env.action_space
        self.observation_space = env.observation_space
        self.observation_space.shape = (84,84,4)
        self.framebuffer = []
        #self.env.unwrapped.frameskip = 4
    #Take a n steps in the environment where n = skipframes*stackframes.
    #The same action is used for all the steps
    #If a step is 'done' repeat the last frame to full up the state.
    #returns a preprocessed state, (normalized)
    
    def reset(self):
        state = self.env.reset()
        state = preprocess(state)
        state = np.stack([state] * 4, axis=2)
        self.framebuffer = state
        self.adversialattack(False)
        return self.framebuffer
    def step(self,action):
        state, reward, done, info = self.env.step(action)
        state = preprocess(state)
        self.framebuffer = np.append(self.framebuffer[:,:,1:], np.expand_dims(state, 2), axis=2)
        if self.clipreward:
            reward=np.sign(reward)
        self.adversialattack()
        return self.framebuffer,reward,done,info 

    def render(self):
        self.env.render()
        
    def adversialattack(self,onlylast=True):
        if not onlylast:
            for x in np.swapaxes(self.framebuffer,0,2):
                print(x.shape)
                for y in range(pixelwhitening):
                    