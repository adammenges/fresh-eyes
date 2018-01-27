from flask import Flask
from collections import defaultdict, OrderedDict
import time
from functools import (reduce, wraps)
import re
import os
import logging
from threading import RLock

# ML
import tensorflow as tf
from keras.backend import learning_phase
from keras.losses import get as get_keras_loss
from keras.metrics import categorical_accuracy
from keras.layers import Input
import numpy as np

app = Flask(__name__)

@app.route("/getPrediction")
def getPrediction():
	return "Prediction!"
