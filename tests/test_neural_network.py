"""
Tests directly related to the class _NN and it's children.

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

# TODO relative imports
from aglaia import NN
from aglaia.aglaia import _NN
from aglaia.wrappers import _ONN, OMNN, OANN
from aglaia.utils import InputError


def hidden_layer_sizes(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hidden_layer_sizes = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hidden_layer_sizes = [4,5])
    C(hidden_layer_sizes = (4,5,6,7))
    C(hidden_layer_sizes = [4.0])

    # This should be caught
    catch([])
    catch([4.2])
    catch(None)
    catch(4)
    catch([0])

def l1_reg(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(l1_reg = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(l1_reg = 0.1)
    C(l1_reg = 1)
    C(l1_reg = 0)

    # This should be caught
    catch(-0.1)
    catch(None)
    catch([0])

def l2_reg(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(l2_reg = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(l2_reg = 0.1)
    C(l2_reg = 1)
    C(l2_reg = 1)

    # This should be caught
    catch(-0.1)
    catch(None)
    catch([0])

def batch_size(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(batch_size = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(batch_size = 2)
    C(batch_size = 2.0)
    C(batch_size = "auto")

    # This should be caught
    catch(1)
    catch(-2)
    catch("x")
    catch(4.2)
    catch(None)

def learning_rate(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(learning_rate = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(learning_rate = 0.1)

    # This should be caught
    catch(0.0)
    catch(-0.1)
    catch(None)

def iterations(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(iterations = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(iterations = 1)
    C(iterations = 1.0)

    # This should be caught
    catch(-2)
    catch(4.2)
    catch(None)

def tf_dtype(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(tf_dtype = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(tf_dtype = 64)
    C(tf_dtype = tf.float64)
    C(tf_dtype = 32)
    C(tf_dtype = tf.float32)
    C(tf_dtype = 16)
    C(tf_dtype = tf.float16)

    # This should be caught
    catch(8)
    catch(float)
    catch(None)

def hl(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hl1 = s)
            C(hl2 = s)
            C(hl3 = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hl1 = 1)
    C(hl1 = 1.0)
    C(hl2 = 1)
    C(hl2 = 1.0)
    C(hl3 = 1)
    C(hl3 = 1.0)

    # This should be caught
    catch(0)
    catch(4.2)
    catch(None)
    catch(-1)


def representations():
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            OMNN(representation = s)
            OANN(representation = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    OMNN(representation = "unsorted_coulomb_matrix")
    OMNN(representation = "sorted_couLomb_matrix")
    OMNN(representation = "bag_of_bOnds")
    OMNN(representation = "slAtm")
    OANN(representation = "atomic_coulomb_matrix")
    OANN(representation = "slAtm")

    # This should be caught
    catch("none")
    catch(4.2)
    catch(None)
    catch(-1)

def test_input():
    # Additional test that inheritance is ok
    for C in _NN, NN, _ONN, OMNN, OANN:
        hidden_layer_sizes(C)
        l1_reg(C)
        l2_reg(C)
        batch_size(C)
        learning_rate(C)
        iterations(C)
        tf_dtype(C)

    for C in _ONN, OMNN, OANN:
        hl(C)

    representations()


def test_NN():

    # Simple example of fitting a quadratic function
    estimator = NN(hidden_layer_sizes=(5, 5, 5), learning_rate=0.01, iterations=5000, l2_reg = 0, tf_dtype = 32, scoring_function="rmse")
    x = np.arange(-2.0, 2.0, 0.05)[:,None]
    y = (x ** 3).ravel()

    estimator.fit(x, y)
    y_pred = estimator.predict(x)

    # set matplotlib to be interactive so the plots wont show
    # TODO should probably set agg backend instead since this distorts the console
    plt.ion()

    # Cost plot
    estimator.plot_cost()
    estimator.correlation_plot(y_pred, y)


def test_minimal_OANN():
    m = OANN(representation='slatm', iterations = 10)
    filenames = glob.glob("../data/qm7/*.xyz")
    random.shuffle(filenames)
    m.generate_compounds(filenames[:2])

    # one property per atom
    d = {}
    for i, c in enumerate(m.compounds):
        x = []
        y = []
        for j in range(len(c.nuclear_charges)):
            y.append(np.random.random())
            x.append(j)
        d[i] = x[:], y[:]


    m.set_properties(d)
    m.fit([0,1])

    # one property per atom pair
    d = {}
    for i, c in enumerate(m.compounds):
        x = []
        y = []
        for j in range(len(c.nuclear_charges)):
            for k in range(j+1, len(c.nuclear_charges)):
                y.append(np.random.random())
                x.append([j,k])
        d[i] = x[:], y[:]

    m.set_properties(d)
    m.fit([0,1])

if __name__ == "__main__":
    test_input()
    test_NN()
    test_minimal_OANN()
