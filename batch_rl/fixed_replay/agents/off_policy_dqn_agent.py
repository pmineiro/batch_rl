# coding=utf-8


"""DQN agent with fixed replay buffer(s) and off-policy considerations."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os

from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from dopamine.agents.dqn import dqn_agent
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class FixedReplayOffPolicyDQNAgent(dqn_agent.DQNAgent):
  """An implementation of the DQN agent with fixed replay buffer(s)."""

  def __init__(self, sess, num_actions, replay_data_dir, replay_suffix=None,
               init_checkpoint_dir=None, **kwargs):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: tf.Session, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      init_checkpoint_dir: str, directory from which initial checkpoint before
        training is loaded if there doesn't exist any checkpoint in the current
        agent directory. If None, no initial checkpoint is loaded.
      **kwargs: Arbitrary keyword arguments.
    """
    assert replay_data_dir is not None
    tf.logging.info(
        'Creating FixedReplayAgent with replay directory: %s', replay_data_dir)
    tf.logging.info('\t init_checkpoint_dir %s', init_checkpoint_dir)
    tf.logging.info('\t replay_suffix %s', replay_suffix)
    # Set replay_log_dir before calling parent's initializer
    self._replay_data_dir = replay_data_dir
    self._replay_suffix = replay_suffix
    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None
    super(FixedReplayOffPolicyDQNAgent, self).__init__(sess, num_actions, **kwargs)

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_observation(observation)
    self.action = self._select_action()
    return self.action


  def end_episode(self, reward):
    assert self.eval_mode, 'Eval mode is not set to be True.'
    super(FixedReplayOffPolicyDQNAgent, self).end_episode(reward)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent."""

    return fixed_replay_buffer.WrappedFixedReplayBuffer(
        data_dir=self._replay_data_dir,
        replay_suffix=self._replay_suffix,
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_reward_op(self):
      off = self.epsilon_eval / self.num_actions
      on = (1 - self.epsilon_eval) + off

      s = self._replay.transition['traj_state']
      a = self._replay.transition['traj_action']
      r = self._replay.transition['traj_reward']
      p = self._replay.transition['traj_states']
      gamma = self._replay.transition['traj_discount']

      flat_s = tf.reshape(s, shape=(-1,) + self.observation_shape)       # b*h x 84 x 84 x 4
      flat_qs = tf.stop_gradient(self.target_convnet(flat_s))            # b*h x num_actions
      flat_qmax = tf.argmax(flat_qs, axis=1)                             # b*h
      flat_pi = tf.one_hot(flat_qmax, depth=self.num_actions, axis=-1, on_value=on, off_value=off)  # b*h x num_actions

      flat_a = tf.reshape(a, (-1,))
      action_mask = tf.one_hot(flat_a, depth=self.num_actions, dtype=tf.bool, on_value=True, off_value=False)

      flat_behavior_probs = tf.boolean_mask(flat_pi, action_mask)                  #b*h
      behavior_probs = tf.reshape(flat_behavior_probs, (-1, self.update_horizon))  #b x h
      importance_weights = behavior_probs / p                                      #b x h
      w = tf.math.cumprod(importance_weights, axis=1)                              #b x h
      return tf.reduce_sum(gamma * w * r, axis=1)                                  #b


  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """

    # TODO(nikosk): Code seems correct for our case.
    # TODO(nikosk): Delete and use base version once verified

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph)
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)



  def _build_target_q_op(self):
    # TODO: include actual trajectory length in the transition so we don't use the wrong cumulative_gamma
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension.
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs.q_values, 1)
    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).

    r = self._build_reward_op()
    return r + self.cumulative_gamma * replay_next_qt_max * (
        1. - tf.cast(self._replay.terminals, tf.float32))


  def _build_train_op(self):
    # TODO: verify it's correct
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """

    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.compat.v1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))
