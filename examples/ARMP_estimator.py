"""This example shows how to use the ARMP estimator from aglaia. It shows how to create the estimator, how to train it,
how to score the performance of it and how to use it to make predictions. It uses a tiny data set with 100 molecules."""

from aglaia import aglaia
import joblib
from sklearn import model_selection as modsel

## ------------- ** Loading the data ** ---------------

data = joblib.load("../data/local_slatm_ch4cn_light.bz")

descriptor = data["descriptor"]
zs = data["zs"]
energies = data["energies"]

## ------------- ** Fitting an estimator ** ---------------

# Creating the estimator
estimator = aglaia.ARMP(hidden_layer_sizes = (5, 3, 2), l1_reg = 0.0001, iterations = 100, learning_rate = 0.01)

# Splitting the data set into training and test set
descriptor_train, descriptor_test, zs_train, zs_test, energies_train, energies_test = \
    modsel.train_test_split(descriptor, zs, energies, test_size=0.3)

# Fitting the model
estimator.fit([descriptor_train, zs_train], energies_train)



## ------------ ** Scoring the estimator ** --------------

score = estimator.score([descriptor_test, zs_test], energies_test)

print("The score is: %s" % (score))



## ------------ ** Using the estimator to predict properties ** ------------

predicted_energies = estimator.predict([descriptor_test, zs_test])



## ------------ ** Plotting the cost and correlation plot** ------------

estimator.plot_cost()
estimator.correlation_plot(predicted_energies, energies_test)
