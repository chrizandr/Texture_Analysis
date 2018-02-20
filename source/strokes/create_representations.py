"""Find representations for clusters."""

import numpy as np
from keras.models import load_model


ENCODER = "autoencoder.hd5"
CLUSTER = ""
autoencoder = load_model(ENCODER)
