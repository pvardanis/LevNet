'''
Update global variables
    
    console: print results depending on platform (command prompt or Jupyter).
    tensorboard: True if use tensorboard, else False.
'''
import os

def init():
    global console, tensorboard, tpu
    console = None
    tensorboard = None
    tpu = None

def cls():
    os.system('clear' if os.name=='posix' else 'cls')