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

"""Logged Replay Buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from concurrent import futures
from dopamine.replay_memory import circular_replay_buffer, off_policy_replay_buffer
from dopamine.replay_memory.circular_replay_buffer import STORE_FILENAME_PREFIX, ReplayElement
from dopamine.replay_memory.sum_tree import SumTree

import numpy as np
import tensorflow.compat.v1 as tf

import gin
gfile = tf.gfile

# a global for now; use gin later
class PrioritizedFixedReplayBuffer(object):
  """Object composed of a list of OutofGraphReplayBuffers."""

  def __init__(self, data_dir, replay_suffix, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Initialize the PrioritizedFixedReplayBuffer class.

    Args:
      data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      *args: Arbitrary extra arguments.
      **kwargs: Arbitrary keyword arguments.
    """
    self._args = args
    self._kwargs = kwargs
    if self._kwargs['extra_storage_types'] is None:
        self._kwargs['extra_storage_types'] = []
    self._kwargs['extra_storage_types'].append(ReplayElement('prob', [], np.float32))
    self._data_dir = data_dir
    self._loaded_buffers = False
    self.add_count = np.array(0)
    self._replay_suffix = replay_suffix
    while not self._loaded_buffers:
      if replay_suffix:
        assert replay_suffix >= 0, 'Please pass a non-negative replay suffix'
        self.load_single_buffer(replay_suffix)
      else:
        self._load_replay_buffers(num_buffers=1)

  def load_single_buffer(self, suffix):
    """Load a single replay buffer."""
    replay_buffer = self._load_buffer(suffix)
    if replay_buffer is not None:
      self._replay_buffers = [replay_buffer]
      self.add_count = replay_buffer.add_count
      self._num_replay_buffers = 1
      self._loaded_buffers = True

  def _load_buffer(self, suffix):
    """Loads a OutOfGraphReplayBuffer replay buffer."""
    try:
      rb = off_policy_replay_buffer.OutOfGraphOffPolicyReplayBuffer(*self._args, **self._kwargs)
      rb.load(self._data_dir, suffix)
      rb._sum_tree = SumTree(rb._replay_capacity)
      assert rb._replay_capacity > rb._update_horizon
      for start in range(rb._replay_capacity - rb._update_horizon):
          end = start + rb._update_horizon
          trajectory_terminals = rb._store['terminal'][start:end]
          is_terminal_transition = trajectory_terminals.any()
          if not is_terminal_transition:
            trajectory_length = rb._update_horizon
          else:
            # np.argmax of a bool array returns the index of the first True.
            trajectory_length = np.argmax(trajectory_terminals.astype(np.bool),
                                          0) + 1

          end = start + trajectory_length
          probs = rb._store['prob'][start:end]
          cp = np.prod(probs)
          if cp > 0:
              prio = np.sqrt(1.0 / max(1e-6, cp))
              # https://en.wikipedia.org/wiki/Square_root_biased_sampling
              rb._sum_tree.set(start, prio)

      tf.logging.info('Loaded replay buffer ckpt {} from {}'.format(
          suffix, self._data_dir))
      # pytype: enable=attribute-error
      return rb
    except tf.errors.NotFoundError:
      # raise
      return None

  def _load_replay_buffers(self, num_buffers=None):
    """Loads multiple checkpoints into a list of replay buffers."""
    if not self._loaded_buffers:  # pytype: disable=attribute-error
      ckpts = gfile.ListDirectory(self._data_dir)  # pytype: disable=attribute-error
      # Assumes that the checkpoints are saved in a format CKPT_NAME.{SUFFIX}.gz
      ckpt_counters = collections.Counter(
          [name.split('.')[-2] for name in ckpts])
      # Should contain the files for add_count, action, observation, reward,
      # terminal and invalid_range
      ckpt_suffixes = [x for x in ckpt_counters if ckpt_counters[x] in [6, 7]]
      if num_buffers is not None:
        ckpt_suffixes = np.random.choice(
            ckpt_suffixes, num_buffers, replace=False)
      self._replay_buffers = []
      # Load the replay buffers in parallel
      with futures.ThreadPoolExecutor(
          max_workers=num_buffers) as thread_pool_executor:
        replay_futures = [thread_pool_executor.submit(
            self._load_buffer, suffix) for suffix in ckpt_suffixes]
      for f in replay_futures:
        replay_buffer = f.result()
        if replay_buffer is not None:
          self._replay_buffers.append(replay_buffer)
          self.add_count = max(replay_buffer.add_count, self.add_count)
      self._num_replay_buffers = len(self._replay_buffers)
      if self._num_replay_buffers:
        self._loaded_buffers = True

  def get_transition_elements(self, batch_size=None):
    rb = self._replay_buffers[0]
    telem = rb.get_transition_elements(batch_size)
    batch_size = rb._batch_size if batch_size is None else batch_size
    telem.append(ReplayElement('sample_probs', (batch_size,), np.float32))
    return telem

  def sample_transition_batch(self, batch_size=None, indices=None):
    assert indices is None
    for _ in range(10):
      # rarely, i get an exception that an index has not been added (?)
      try:
        buffer_index = np.random.randint(self._num_replay_buffers)
        rb = self._replay_buffers[buffer_index]
        batch_size = rb._batch_size if batch_size is None else batch_size
        indices = rb._sum_tree.stratified_sample(batch_size)
        tb = rb.sample_transition_batch(batch_size=batch_size, indices=indices)
        probs = np.array(rb._sum_tree.get(indices))
        return tb + (probs.astype(np.single), )
      except:
        pass

    # shrug ... just do a uniform sample
    buffer_index = np.random.randint(self._num_replay_buffers)
    rb = self._replay_buffers[buffer_index]
    batch_size = rb._batch_size if batch_size is None else batch_size
    tb = rb.sample_transition_batch(batch_size=batch_size)
    probs = np.ones(batch_size)
    return tb + (probs.astype(np.single),)

  def load(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

  def reload_buffer(self, num_buffers=None):
    self._loaded_buffers = False
    self._load_replay_buffers(num_buffers)

  def save(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

  def add(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedPrioritizedFixedReplayBuffer(off_policy_replay_buffer.WrappedOffPolicyReplayBuffer):
  """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism."""

  def __init__(self,
               data_dir,
               replay_suffix,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               wrapped_memory=None,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32):
    """Initializes WrappedPrioritizedFixedReplayBuffer."""

    memory = PrioritizedFixedReplayBuffer(
        data_dir, replay_suffix, observation_shape, stack_size, replay_capacity,
        batch_size, update_horizon, gamma, max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype)

    super(WrappedPrioritizedFixedReplayBuffer, self).__init__(
        observation_shape,
        stack_size,
        use_staging=use_staging,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        wrapped_memory=memory,
        max_sample_attempts=max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)
