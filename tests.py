############################################################
# Overview
#
# A few tests to confirm functions' results look right
############################################################

import numpy as np
import functions as ds

def make_test_function(f):
    def out():
        
        # defaults run
        X, Y = f()
        T0 = X.shape == (200, 10)
        
        # Confirm shape change as expected
        X, Y = f(100, 5)
        T1 = X.shape == (100, 5)
        T2 = Y.shape == (100,)
        
        X, Y = f(501, 3)
        T3 = X.shape == (501, 3)
        T4 = Y.shape == (501,)
        
        # y contains a mix of 0s and 1s.
        T5 = np.mean(Y) > 0 and np.mean(Y) < 1
        T6 = np.min(Y) == 0
        T7 = np.max(Y) == 1
        
        return T0 and T1 and T2 and T3 and T4 and T5  and T6 and T7
    return out

test_friedman_1 = make_test_function(f = ds.friedman_1)
test_friedman_2 = make_test_function(f = ds.friedman_2)
test_friedman_3 = make_test_function(f = ds.friedman_3)
test_friedman_4 = make_test_function(f = ds.friedman_4)
test_friedman_5 = make_test_function(f = ds.friedman_5)

def test_waveform():
    # defaults run
    X, Y = ds.waveform()
    T0 = X.shape == (200, 4)
    
    # Confirm shape change as expected
    X, Y = ds.waveform(100)
    T1 = X.shape == (100, 4)
    T2 = Y.shape == (100,)
    
    X, Y = ds.waveform(301)
    T3 = X.shape == (301, 4)
    T4 = Y.shape == (301,)
    
    # y contains a mix of 0s, 1s, 2s.
    T5 = np.mean(Y) > 1 and np.mean(Y) < 1.1
    T6 = np.min(Y) == 0
    T7 = np.max(Y) == 2
    
    return T0 and T1 and T2 and T3 and T4 and T5 and T6 and T7

test_friedman_1()
test_friedman_2()
test_friedman_3()
test_friedman_4()
test_friedman_5()
test_waveform()
