import tensorflow as tf
import numpy as np
from policies import create_network

class A3CNetwork(object):

    def __init__(self, session, env, scope="global", GlobalNet=None, lr=1e-4, decay=0.99, eps=1e-6, val_w = 0.5, entropy_w = 0.001, network_type="ff"):
        self.sess = session
        self.action_n = env.action_space.n
        self.state_shape = env.observation_space.shape
        self.val_w = val_w
        self.entropy_w = entropy_w
        self.ppo_clip = 0.2
        self.GlobalNet = GlobalNet

        self.tf_states = tf.placeholder(tf.float32, (None,) + self.state_shape, name="states")
        self.tf_actions = tf.placeholder(tf.int32, shape=[None], name="actions")

        if scope == "global":
            self.global_step = tf.Variable(0,trainable=False, name='global_step')
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr,decay=decay)

            with tf.variable_scope("ppo_model"):
                self.ppo_pi, self.ppo_quality = create_network(self.tf_states, self.action_n, network_type, 1)
                mask = tf.range(0, tf.shape(self.ppo_pi)[0]) * tf.shape(self.ppo_pi[1]) + self.tf_actions
                self.ppo_action_pi = tf.gather(tf.reshape(self.ppo_pi, [-1]), mask)
                self.log_ppo_a_pi = tf.log(tf.clip_by_value(self.ppo_action_pi,eps,1-eps))

            self.ppo_params = tf.trainable_variables("ppo_model")

        with tf.variable_scope("net_" + scope):
            self.pi, self.values = create_network(self.tf_states, self.action_n, network_type, 1)

        self.tf_trainable = tf.trainable_variables("net_" + scope)

        if scope != "global":
            self.tf_targets = tf.placeholder(tf.float32, shape=[None], name="targets")
            self.tf_advantages =  tf.placeholder(tf.float32, shape=[None], name="advantages")
            self.tf_ppo_action_pi_input = tf.placeholder(tf.float32, shape = [None], name="ppo_pi_input")
            mask = tf.range(0, tf.shape(self.pi)[0]) * tf.shape(self.pi)[1] + self.tf_actions

            self.log_pi = tf.log(tf.clip_by_value(self.pi, eps, 1-eps)) # prevents log(0)/log(1)->0

            self.action_log_pi = tf.gather(tf.reshape(self.log_pi, [-1]), mask)
            self.action_pi = tf.gather(tf.reshape(self.pi, [-1]), mask)

            with tf.name_scope("losses"):
                with tf.name_scope("entropy_loss"):
                    self.entropy_loss = tf.reduce_mean(tf.multiply(self.pi, -self.log_pi))
                with tf.name_scope("policy_loss"):
                    self.pi_loss = -tf.reduce_mean(self.action_log_pi*self.tf_advantages)
                    #ratio = tf.exp(self.action_log_pi - tf.log(tf.clip_by_value(self.tf_ppo_action_pi_input,eps,1-eps)))
                    #ratio = tf.divide(self.action_pi,self.tf_ppo_action_pi_input+eps)
                    #ppo_loss1 = self.tf_advantages * ratio
                    #ppo_loss2 = self.tf_advantages * tf.clip_by_value(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
                    #self.pi_loss = -tf.reduce_mean(tf.minimum(ppo_loss1,ppo_loss2))

                with tf.name_scope("value_loss"):
                    self.value_loss = tf.reduce_mean(tf.square(self.tf_targets-self.values))
                with tf.name_scope("merged_loss"):
                    self.loss = self.pi_loss - self.entropy_loss * self.entropy_w + self.value_loss * self.val_w

            with tf.name_scope("summary"):
                tf.summary.scalar("entropy", self.entropy_loss)
                tf.summary.scalar("policy_loss", self.pi_loss)
                tf.summary.scalar("value_loss", self.value_loss)
                tf.summary.scalar("loss", self.loss)
                self.summary_op = tf.summary.merge_all()

            with tf.name_scope("update_global"):
                grads = tf.gradients(self.loss, self.tf_trainable)
                clipped_grads,_ = tf.clip_by_global_norm(grads,40)
                grads_vars = list(zip(clipped_grads,GlobalNet.tf_trainable))
                self.train_op = GlobalNet.optimizer.apply_gradients(grads_vars, global_step=GlobalNet.global_step)

            with tf.name_scope("update_local_net"):
                self.update_target_net = [local_params.assign(global_params) for local_params, global_params in
                                          zip(self.tf_trainable, GlobalNet.tf_trainable)]

            with tf.name_scope("update_ppo_net"):
                self.update_ppo_net = [ppo_params.assign(global_params) for ppo_params, global_params in zip(GlobalNet.ppo_params, GlobalNet.tf_trainable)]

    def get_pi(self,states):
        states = self.prep_states(states)
        pi = self.sess.run(self.pi,{self.tf_states:states})
        return pi
    def get_ppo_pi(self,states):
        states = self.prep_states(states)
        pi = self.sess.run(self.ppo_pi,{self.tf_states:states})
        return pi
    def get_ppo_pi_for_actions(self,states,actions):
        states = self.prep_states(states)
        action_pi = self.sess.run(self.ppo_action_pi,{self.tf_states:states, self.tf_actions:actions})
        return action_pi

    def update_step(self,states,actions, targets, advantages,ppo_action_pi, summary = False):
        states = self.prep_states(states)
        step = 0
        feeddict = {self.tf_states:states,self.tf_actions:actions,self.tf_targets:targets,self.tf_advantages:advantages,self.GlobalNet.tf_states:states, self.GlobalNet.tf_actions:actions,self.tf_ppo_action_pi_input:ppo_action_pi}
        if summary:
            summary, _, step = self.sess.run([self.summary_op, self.train_op, self.GlobalNet.global_step], feeddict)
            return summary, step
        _, step = self.sess.run([self.train_op, self.GlobalNet.global_step], feeddict)
        return None, step

    def update_target(self):
        self.sess.run(self.update_target_net)

    def update_ppo(self):
        self.sess.run(self.update_ppo_net)

    def get_action(self,states):
        states = self.prep_states(states)
        pi = self.get_pi(states)[0]
        action = np.random.choice(self.action_n, 1, p=pi)[0]
        return action

    def get_values(self, states):
        states = self.prep_states(states)
        values = self.sess.run(self.values,{self.tf_states:states})
        return values

    def prep_states(self,states):
        
        if len(np.array(states).shape) < len(self.state_shape)+1:
            states = np.expand_dims(states, 0)
        if type(states[0][0][0][0]) == np.uint8:
            states = np.divide(states,255)
            
        return states
