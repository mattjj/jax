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

"""Runs each example's `--check` mode, so the examples can't rot silently.

The examples are run as subprocesses because each one configures the number of
simulated devices before the backend is initialized, which can only be done
once per process. A small device count is used on purpose: XLA's CPU backend
needs one thread per participating device to complete a collective, so asking
for more devices than the test machine has cores can stall.
"""

import os
import subprocess
import sys

from absl.testing import absltest
from absl.testing import parameterized

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICES = '2'


def run(script, *args):
  env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get('PYTHONPATH', ''))
  return subprocess.run([sys.executable, os.path.join(HERE, script), *args],
                        capture_output=True, text=True, env=env, timeout=900)


class ExamplesTest(parameterized.TestCase):

  @parameterized.parameters(
      ('nanolm.py', ('--check', '--offline', '--devices', DEVICES, '--mesh', '2,1')),
      ('nanolm.py', ('--check', '--offline', '--devices', DEVICES, '--mesh', '1,2')),
      ('moe.py', ('--check', '--devices', DEVICES, '--mesh', '2')),
      ('sample.py', ('--check', '--offline', '--devices', DEVICES, '--mesh', '2,1',
                     '--train-steps', '2')),
      ('fsdp_pipeline.py', ('--check', '--offline', '--devices', DEVICES)),
  )
  def test_check(self, script, args):
    p = run(script, *args)
    self.assertEqual(p.returncode, 0, msg=p.stdout + p.stderr)
    self.assertIn('check:', p.stdout)

  @parameterized.parameters(
      ('nanolm.py', ('--steps', '3', '--offline', '--devices', DEVICES, '--mesh', '2,1')),
      ('moe.py', ('--steps', '3', '--devices', DEVICES, '--mesh', '2')),
  )
  def test_trains(self, script, args):
    p = run(script, *args)
    self.assertEqual(p.returncode, 0, msg=p.stdout + p.stderr)
    self.assertIn('loss', p.stdout)


if __name__ == '__main__':
  absltest.main()
