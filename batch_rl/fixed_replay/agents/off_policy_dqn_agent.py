# coding=utf-8


"""DQN agent with fixed replay buffer(s) and off-policy considerations."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os

from batch_rl.fixed_replay.agents import empiricalpdis as emppdis
from batch_rl.fixed_replay.agents import incrementaliwlb as incriwlb
from batch_rl.fixed_replay.agents import incrementaliwlbmoment as iiwlbmom
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
    nbatches = kwargs.pop('nbatches')
    self.coverage = kwargs.pop('coverage')
    self.rmin = float(kwargs.pop('rmin'))
    self.coverage_decay = kwargs.pop('coverage_decay')
    self.sarsa = kwargs.pop('sarsa')
    self.uniform_propensities = kwargs.pop('uniform_propensities')
    self.qlambda = kwargs.pop('qlambda')
    empirical_pdis = kwargs.pop('empirical_pdis')
    moment_constraint = kwargs.pop('moment_constraint')
    self.adjust_lr = kwargs.pop('adjust_lr')
    super(FixedReplayOffPolicyDQNAgent, self).__init__(sess, num_actions, **kwargs)
    if empirical_pdis:
        self.iiwlbmom = emppdis.EmpiricalPdis()
    elif moment_constraint:
        self.iiwlbmom = iiwlbmom.IncrementalIwLbMoment(coverage=self.coverage, nbatches=nbatches)
    else:
        self.iiwlbmom = incriwlb.IncrementalIwLb(coverage=self.coverage, nbatches=nbatches)
    self.iteration_count = 0
    self.iteration_end_hook()

  def compute_coverage(self):
      if self.coverage_decay:
          from math import pi

          t = self.iteration_count
          scalefac = 6 / pi**2

          new_coverage = 1.0 - (1.0 - self.coverage) * scalefac / t**2
          tf.logging.info('\t setting coverage to %g', new_coverage)

          return new_coverage
      else:
          return self.coverage

  def iteration_end_hook(self):
      self.iteration_count += 1
      self.iiwlbmom.coverage = self.compute_coverage()

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
      if self.qlambda:
          p = tf.constant(1.0, shape=r.shape, dtype=r.dtype)
          off, on = 0.0, 1.0
      elif self.uniform_propensities:
          p = tf.constant(1.0 / self.num_actions, shape=r.shape, dtype=r.dtype)
      else:
          p = self._replay.transition['traj_prob']
      gamma = self._replay.transition['traj_discount']

      state_shape = self.observation_shape + (self.stack_size,)
      flat_s = tf.reshape(s, shape=(-1,) + state_shape)                 # b*h x 84 x 84 x 4
      flat_qs = tf.stop_gradient(self.target_convnet(flat_s).q_values)  # b*h x num_actions
      flat_qmax = tf.argmax(flat_qs, axis=1)                            # b*h
      flat_pi = tf.one_hot(flat_qmax, depth=self.num_actions, axis=-1, on_value=on, off_value=off)  # b*h x num_actions

      flat_a = tf.reshape(a, (-1,))
      action_mask = tf.one_hot(flat_a, depth=self.num_actions, dtype=tf.bool, on_value=True, off_value=False)
      flat_behavior_probs = tf.boolean_mask(flat_pi, action_mask)                  #b*h
      behavior_probs = tf.reshape(flat_behavior_probs, (-1, self.update_horizon))  #b x h
      flassimp = behavior_probs / p                                      #b x h
      # NB: tensorflow sucks
      def assign_ones(w):
          w[:, 0] = 1
          return w
      importance_weights = tf.numpy_function(assign_ones, [ flassimp ], tf.float32)
      w = tf.math.cumprod(importance_weights, axis=1)                              #b x h
      if self.rmin == 0:
          q = tf.numpy_function(lambda *args: self.iiwlbmom.tfhook(*args), [ gamma, w, r ], tf.float32)
      else:
          q = tf.numpy_function(lambda *args: self.iiwlbmom.tfhook(*args), [ gamma, w, r - self.rmin ], tf.float32)

      if self.summary_writer is not None:
        duals = tf.numpy_function(lambda *args: self.iiwlbmom.dualstfhook(*args), [ ], tf.float32)
        with tf.compat.v1.variable_scope('Duals'):
          tf.compat.v1.summary.scalar('v', duals[0])
          tf.compat.v1.summary.scalar('alpha', duals[1])
          tf.compat.v1.summary.scalar('beta', duals[2])
          tf.compat.v1.summary.scalar('kappa', duals[3])

      return q * tf.reduce_sum(gamma * w * r, axis=1)                              #b

  def _build_target_q_op(self):
    # TODO: include actual trajectory length in the transition so we don't use the wrong cumulative_gamma
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    if self.sarsa:
        flat_next_a = tf.reshape(self._replay.transition['next_action'], (-1,))
        next_a_mask = tf.reshape(tf.one_hot(flat_next_a, depth=self.num_actions, axis=-1, on_value=True, off_value=False), (-1,))
        flat_next_q = tf.reshape(self._replay_next_target_net_outputs.q_values, (-1,))
        flat_target = tf.boolean_mask(flat_next_q, next_a_mask)
        replay_next_qt_max = tf.reshape(flat_target, (-1, 1))
    else:
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
    loss = tf.reduce_mean(tf.compat.v1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE))
    if self.adjust_lr is not None:
        scalefac = tf.stop_gradient(tf.clip_by_value(loss, self.adjust_lr, 10000))
        scaleloss = loss * (self.adjust_lr / scalefac)
    else:
        scaleloss = loss
    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        tf.compat.v1.summary.scalar('HuberLoss', scaleloss)
    return self.optimizer.minimize(scaleloss)
