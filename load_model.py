
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# 🔧 rebuild model (exact copy)
def build_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    return model

# load weights
model = build_model()
model.load_weights("model.weights.h5")

print("✅ Model rebuilt and weights loaded")

# test
dummy = np.random.rand(1,128,128,3)
pred = model.predict(dummy)

import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# 🔧 rebuild model (exact copy)
def build_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    return model

# load weights
model = build_model()
model.load_weights("model.weights.h5")

print("✅ Model rebuilt and weights loaded")

# test
dummy = np.random.rand(1,128,128,3)
pred = model.predict(dummy)

print("Prediction shape:", pred.shape)