#  Importing essential libraries
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians datasetdiabetes
f_fires = np.loadtxt("forestfires.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = f_fires[:,2:30]
Y = f_fires[:,30]


# create model
model = Sequential()
model.add(Dense(12, input_dim=30, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Visualize training history

# list all data in history
model.history.history.keys()


