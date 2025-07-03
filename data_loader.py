# === 3.1 Edge Layer ===
# Signal acquisition and lightweight preprocessing

import wfdb  # Library to load/download PhysioNet signal records
import os
import numpy as np
from scipy.signal import butter, lfilter  # Signal filtering functions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential