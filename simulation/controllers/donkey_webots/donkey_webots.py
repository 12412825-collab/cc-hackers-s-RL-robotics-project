"""Webots controller entry point for the DonkeyCar application.

Set the Webots Robot controller field to ``donkey_webots``. Webots starts this
script with its controller Python environment; the script then runs the normal
DonkeyCar ``manage.py drive`` command.
"""

import os
import runpy
import sys


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

model_path = os.environ.get('DONKEY_MODEL_PATH')
sys.argv = [os.path.join(PROJECT_ROOT, 'manage.py'), 'drive']
if model_path:
    sys.argv.append(f'--model={model_path}')

runpy.run_path(os.path.join(PROJECT_ROOT, 'manage.py'), run_name='__main__')
