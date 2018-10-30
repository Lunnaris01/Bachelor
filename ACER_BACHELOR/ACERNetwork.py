import tensorflow as tf
import numpy as np
from policies import create_network


class ACERNetwork(object):

    def __init__(self, session, env, scope="global", GlobalNet=None, lr=2e-4, decay=0.99, eps=1e-6,
                 value_w=0.25, entropy_w=0.01, network_type="cnn"):
        self.sess = session
        self.action_n = env.action_space.n
        self.state_shape = env.observation_space.shape
        self.val_w = value_w
        self.entropy_w = entropy_w
        self.trunc_val = 10.0
        self.GlobalNet = GlobalNet

        self.tf_states = tf.placeholder(tf.float32, (None,) + self.state_shape, name="states")
        self.tf_actions = tf.placeholder(tf.int32, shape=[None], name="actions")

        if scope == "global":
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=decay)

        with tf.variable_scope("net_" + scope):
            self.pi, self.quality = create_network(self.tf_states, self.action_n, network_type, self.action_n)
        mask = tf.range(0, tf.shape(self.pi)[0]) * tf.shape(self.pi)[1] + self.tf_actions
        self.action_quality = tf.gather(tf.reshape(self.quality, [-1]), mask)
        with tf.name_scope("values"):
            self.values = tf.reduce_sum(tf.multiply(self.quality, tf.stop_gradient(self.pi)), axis=-1)


        self.tf_trainable = tf.trainable_variables("net_" + scope)

        if scope != "global":
            self.tf_q_retrace_targets = tf.placeholder(tf.float32, shape=[None], name="retrace_targets")
            self.tf_importance_weights = tf.placeholder(tf.float32, shape=[None, self.action_n], name="Importance_weights")

            self.advantages = self.tf_q_retrace_targets - self.values
            self.log_pi = tf.log(tf.clip_by_value(self.pi, eps, 1 - eps))
            self.action_prob = tf.gather(tf.reshape(self.pi, [-1]), mask)
            self.action_log_prob = tf.gather(tf.reshape(self.log_pi, [-1]), mask)
            self.action_importance_weights = tf.gather(tf.reshape(self.tf_importance_weights, [-1]), mask)

            with tf.name_scope("losses"):
                with tf.name_scope("entropy"):
                    self.entropy = tf.reduce_sum(
                        tf.multiply(self.pi, - self.log_pi * self.entropy_w))
                with tf.name_scope("policy_loss"):
                    pi_loss = -tf.reduce_mean(
                        tf.multiply(self.action_log_prob, tf.stop_gradient(
                            self.advantages * tf.minimum(self.trunc_val, self.action_importance_weights))))
                    adv_bias_correction = (self.quality - tf.reshape(self.values, [-1, 1]))
                    gain_bc = tf.reduce_sum(self.log_pi * tf.stop_gradient(adv_bias_correction * tf.nn.relu(
                        1.0 - (self.trunc_val / (self.tf_importance_weights + eps))) * self.pi), axis=1)
                    loss_bc = -tf.reduce_mean(gain_bc)
                    self.policy_loss = pi_loss + loss_bc

                with tf.name_scope("value_loss"):
                    self.value_loss = tf.reduce_mean(
                        tf.square(self.tf_q_retrace_targets - self.action_quality))

                with tf.name_scope("merged_loss"):
                    self.loss = self.policy_loss - self.entropy * self.entropy_w + self.value_loss * self.val_w

            with tf.name_scope("summary"):
                tf.summary.scalar("entropy", self.entropy)
                tf.summary.scalar("policy_loss", self.policy_loss)
                tf.summary.scalar("value_loss", self.value_loss)
                tf.summary.scalar("loss", self.loss)
                self.summary_op = tf.summary.merge_all()

            with tf.name_scope("update_global"):
                grads = tf.gradients(self.loss, self.tf_trainable)
                clipped_grads, _ = tf.clip_by_global_norm(grads, 10)
                grads_vars = list(zip(clipped_grads, GlobalNet.tf_trainable))
                self.train_op = GlobalNet.optimizer.apply_gradients(grads_vars, global_step=GlobalNet.global_step)

            with tf.name_scope("sync_target_net"):
                self.update_target_net = [local_params.assign(global_params) for local_params, global_params in
                                          zip(self.tf_trainable, GlobalNet.tf_trainable)]

    def get_retrace_values(self, states, actions):
        states = self.prep_states(states)
        vals = self.sess.run([self.GlobalNet.pi, self.action_quality, self.values],
                             {self.tf_states: states, self.tf_actions: actions, self.GlobalNet.tf_states: states})
        return vals

    def update_target(self):
        self.sess.run(self.update_target_net)

    def update_step(self, states, actions, targets, importance_w, summary=False):
        states = self.prep_states(states)
        feeddict = {self.tf_states: states, self.tf_actions: actions,
                    self.tf_q_retrace_targets: targets, self.tf_importance_weights: importance_w}

        if summary:
            _, summary, step = self.sess.run([self.train_op, self.summary_op, self.GlobalNet.global_step], feeddict)
            return summary, step

        _, step = self.sess.run([self.train_op, self.GlobalNet.global_step], feeddict)
        return None, step

    def get_pi(self, states):
        states = self.prep_states(states)
        pi = self.sess.run(self.pi, {self.tf_states: states})
        return pi

    def get_action_with_probs(self, states):
        states = self.prep_states(states)
        pi = self.get_pi(states)[0]
        action = np.random.choice(self.action_n, 1, p=pi)[0]
        return action, pi
    
    def get_action(self, states):
        action, _ = self.get_action_with_probs(states)
        return action

    def get_values(self, states):
        states = self.prep_states(states)
        values = self.sess.run(self.values, {self.tf_states: states})
        return values

    def prep_states(self, states):
        if len(np.array(states).shape) < len(self.state_shape) + 1:
            states = np.expand_dims(states, 0)
        if type(states[0][0][0][0]) == np.uint8:
            states = np.divide(states, 255)

        return states
