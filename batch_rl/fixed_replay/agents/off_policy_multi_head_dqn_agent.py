# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi Head DQN agent with fixed replay buffer(s)."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from batch_rl.fixed_replay.agents import empiricalpdis as emppdis
from batch_rl.fixed_replay.agents import incrementaliwlb as incriwlb
from batch_rl.fixed_replay.agents import incrementaliwlbmoment as iiwlbmom
from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from batch_rl.multi_head import multi_head_dqn_agent
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class FixedReplayOffPolicyMultiHeadDQNAgent(multi_head_dqn_agent.MultiHeadDQNAgent):
  """MultiHeadDQNAgent with fixed replay buffer(s)."""

  def __init__(self, sess, num_actions, replay_data_dir, replay_suffix=None,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: tf.Session, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      **kwargs: Arbitrary keyword arguments.
    """
    assert replay_data_dir is not None
    tf.logging.info(
        'Creating FixedReplayOffPolicyMultiHeadDQNAgent with replay directory: %s',
        replay_data_dir)
    tf.logging.info('\t replay_suffix %s', replay_suffix)
    # Set replay_log_dir before calling parent's initializer
    self._replay_data_dir = replay_data_dir
    self._replay_suffix = replay_suffix
    nbatches = kwargs.pop('nbatches')
    coverage = kwargs.pop('coverage')
    self.rmin = float(kwargs.pop('rmin'))
    self.sarsa = kwargs.pop('sarsa')
    self.uniform_propensities = kwargs.pop('uniform_propensities')
    self.qlambda = kwargs.pop('qlambda')
    empirical_pdis = kwargs.pop('empirical_pdis')
    moment_constraint = kwargs.pop('moment_constraint')
    super(FixedReplayOffPolicyMultiHeadDQNAgent, self).__init__(
        sess, num_actions, **kwargs)
    if empirical_pdis:
        # TODO
        assert False
        self.iiwlbmom = emppdis.EmpiricalPdis()
    elif moment_constraint:
        self.iiwlbmommulti = iiwlbmom.MultiIncrementalIwLbMoment(coverage=coverage, nbatches=nbatches, nheads=self.num_heads)
    else:
        # TODO
        assert False
        self.iiwlbmom = incriwlb.IncrementalIwLb(coverage=coverage, nbatches=nbatches)

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
    super(FixedReplayOffPolicyMultiHeadDQNAgent, self).end_episode(reward)

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
      flat_qs = tf.stop_gradient(self.target_convnet(flat_s).q_heads)   # b*h x num_actions x num_heads
      flat_qmax = tf.argmax(flat_qs, axis=1)                            # b*h x num_heads
      flat_pi = tf.one_hot(flat_qmax, depth=self.num_actions, axis=1, on_value=on, off_value=off)  # b*h x num_actions x num_heads
      flat_a = tf.reshape(a, (-1,)) # b*h
      action_mask = tf.one_hot(flat_a, depth=self.num_actions, dtype=tf.bool, on_value=True, off_value=False) # b*h x num_actions
      flat_behavior_probs = tf.boolean_mask(flat_pi, action_mask)                  #b*h x num_heads
      behavior_probs = tf.reshape(flat_behavior_probs, (-1, self.update_horizon, self.num_heads))  #b x h x num_heads
      p_heads = tf.expand_dims(p, axis=-1) # b x h x 1
      flassimp = behavior_probs / p_heads  # b x h x num_heads

      # NB: tensorflow sucks
      def assign_ones(w):
          w[:, 0, :] = 1
          return w
      importance_weights = tf.numpy_function(assign_ones, [ flassimp ], tf.float32) # b x h x num_heads
      w = tf.math.cumprod(importance_weights, axis=1) #b x h x num_heads
      if self.rmin == 0:
          q = tf.numpy_function(lambda *args: self.iiwlbmommulti.tfhook(*args), [ gamma, w, r ], tf.float32) # b x num_heads
      else:
          q = tf.numpy_function(lambda *args: self.iiwlbmommulti.tfhook(*args), [ gamma, w, r - self.rmin ], tf.float32) # b x num_heads

      if self.summary_writer is not None:
        duals = tf.numpy_function(lambda *args: self.iiwlbmommulti.dualstfhook(*args), [ ], tf.float32) # 4 x num_heads
        meanduals = tf.reduce_mean(duals, axis=-1)
        with tf.compat.v1.variable_scope('Duals'):
          tf.compat.v1.summary.scalar('v', meanduals[0])
          tf.compat.v1.summary.scalar('alpha', meanduals[1])
          tf.compat.v1.summary.scalar('beta', meanduals[2])
          tf.compat.v1.summary.scalar('kappa', meanduals[3])

      biggamma = tf.expand_dims(gamma, axis=-1) # b x h x 1
      bigr = tf.expand_dims(r, axis=-1) # b x h x 1
      return q * tf.reduce_sum(biggamma * w * bigr, axis=1)

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension for each head.
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs.q_heads, axis=1)
    is_non_terminal = 1. - tf.cast(self._replay.terminals, tf.float32)
    is_non_terminal = tf.expand_dims(is_non_terminal, axis=-1)
    r = self._build_reward_op()
    return r + (
        self.cumulative_gamma * replay_next_qt_max * is_non_terminal)
