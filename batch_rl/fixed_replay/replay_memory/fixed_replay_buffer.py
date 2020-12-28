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

import numpy as np
import tensorflow.compat.v1 as tf

import gin
gfile = tf.gfile

STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX

# a global for now; use gin later
use_off_policy_replay_buffer = True

class FixedReplayBuffer(object):
  """Object composed of a list of OutofGraphReplayBuffers."""

  def __init__(self, data_dir, replay_suffix, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Initialize the FixedReplayBuffer class.

    Args:
      data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      *args: Arbitrary extra arguments.
      **kwargs: Arbitrary keyword arguments.
    """
    self._args = args
    self._kwargs = kwargs
    if use_off_policy_replay_buffer:
        if self._kwargs['extra_storage_types'] is None:
            self._kwargs['extra_storage_types'] = []
        self._kwargs['extra_storage_types'].append(circular_replay_buffer.ReplayElement('prob', [], np.float32))
    self._data_dir = data_dir
    self._loaded_buffers = False
    self.add_count = np.array(0)
    self._replay_suffix = replay_suffix
    self._maxbuffernum = kwargs.pop('maxbuffernum')
    self._stratified_sample = kwargs.pop('stratified_sample')
    self._inorder = kwargs.pop('inorder')
    self._loadcount = 0
    while not self._loaded_buffers:
      if replay_suffix:
        assert replay_suffix >= 0, 'Please pass a non-negative replay suffix'
        self.load_single_buffer(replay_suffix)
      else:
        self._load_replay_buffers(num_buffers=1)
    self._loadcount = 0

  def load_single_buffer(self, suffix):
    """Load a single replay buffer."""
    replay_buffer = self._load_buffer(suffix)
    if replay_buffer is not None:
      self._replay_buffers = [replay_buffer]
      self.add_count = replay_buffer.add_count
      self._num_replay_buffers = 1
      self._loaded_buffers = True
      self._loadcount += 1

  def _load_buffer(self, suffix):
    """Loads a OutOfGraphReplayBuffer replay buffer."""
    try:
      if use_off_policy_replay_buffer:
        replay_buffer = off_policy_replay_buffer.OutOfGraphOffPolicyReplayBuffer(*self._args, subsample_seed=suffix, **self._kwargs)
      else:
        # pytype: disable=attribute-error
        replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
            *self._args, **self._kwargs)
      replay_buffer.load(self._data_dir, suffix)
      tf.logging.info('Loaded replay buffer ckpt {} from {}'.format(
          suffix, self._data_dir))
      # pytype: enable=attribute-error
      return replay_buffer
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
      if self._maxbuffernum is not None:
          ckpt_suffixes = [v for v in ckpt_suffixes if int(v) < self._maxbuffernum]
      if num_buffers is not None:
          if self._inorder:
              minindex = self._loadcount % len(ckpt_suffixes)
              maxindex = (self._loadcount + num_buffers) % len(ckpt_suffixes)
              ckpt_suffixes = list(sorted(ckpt_suffixes, key=int))[minindex:maxindex]
              self._loadcount += 1
          elif self._stratified_sample:
              def get_chunks(lst, n):
                  chunksize = len(lst) // n
                  rv = []
                  chunkcnt = 1
                  for m, v in enumerate(lst):
                      rv.append(v)
                      if len(rv) >= chunksize:
                          yield rv
                          rv = []
                          chunkcnt += 1
                      if chunkcnt >= n:
                          rv = lst[m+1:]
                          break

                  if len(rv):
                      yield rv

              chosen = []
              ckpt_suffixes = list(sorted(ckpt_suffixes, key=int))
              for chunk in get_chunks(ckpt_suffixes, num_buffers):
                  chosen.append(np.random.choice(chunk, 1).item())
              ckpt_suffixes = chosen
          else:
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

  def get_transition_elements(self):
    return self._replay_buffers[0].get_transition_elements()

  def sample_transition_batch(self, batch_size=None, indices=None):
    buffer_index = np.random.randint(self._num_replay_buffers)
    return self._replay_buffers[buffer_index].sample_transition_batch(
        batch_size=batch_size, indices=indices)

  def load(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

  def reload_buffer(self, num_buffers=None):
    self._loaded_buffers = False
    self._load_replay_buffers(num_buffers)

  def save(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

  def add(self, *args, **kwargs):  # pylint: disable=unused-argument
    pass

parent = off_policy_replay_buffer.WrappedOffPolicyReplayBuffer if use_off_policy_replay_buffer else circular_replay_buffer.WrappedReplayBuffer

@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedFixedReplayBuffer(parent):
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
               reward_dtype=np.float32,
               maxbuffernum=None,
               stratified_sample=False,
               subsample_percentage=None,
               inorder=False):
    """Initializes WrappedFixedReplayBuffer."""

    memory = FixedReplayBuffer(
        data_dir, replay_suffix, observation_shape, stack_size, replay_capacity,
        batch_size, update_horizon, gamma, max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        maxbuffernum=maxbuffernum, stratified_sample=stratified_sample,
        subsample_percentage=subsample_percentage, inorder=inorder)

    super(WrappedFixedReplayBuffer, self).__init__(
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
