# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared defaults for running the examples on simulated CPU devices."""

import os


def default_devices(most: int = 8) -> int:
  """How many simulated CPU devices it is safe to ask for.

  All simulated CPU devices are backed by a single thread pool sized to the
  machine's cores, and a collective blocks a thread per participating device
  until every one of them arrives. So asking for more devices than there are
  cores can deadlock rather than merely run slowly, and the default here stays
  at or below the core count. Pass a larger `--devices` explicitly if you know
  your machine can take it.
  """
  try:
    cores = len(os.sched_getaffinity(0))  # respects cgroup limits and taskset
  except AttributeError:  # not available off Linux
    cores = os.cpu_count() or 1
  return max(1, min(most, cores))


def default_mesh(devices: int) -> str:
  """A "data,model" mesh shape for `devices` devices, favoring data."""
  model = 2 if devices >= 4 and devices % 2 == 0 else 1
  return f'{devices // model},{model}'
