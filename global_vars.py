'''
Update global variables
    
    console: print results depending on platform (command prompt or Jupyter).
    tensorboard: True if use tensorboard, else False.
'''
import os

def init():
    global console, tensorboard
    console = None
    tensorboard = None

def cls():
    os.system('clear')
    # os.system('cls' if os.name=='nt' else 'clear')