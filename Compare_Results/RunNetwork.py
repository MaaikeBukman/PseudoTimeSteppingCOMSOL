import numpy as np
from MakePredictions import make_predictions

def run_network(model, network, iter, max_iter, patience):
    CONV_list = []
    for i in range(max_iter):
        conv, _, _ = make_predictions(model, network, iter, i+1)
        CONV_list.append(conv)
        print(f"{i+1}   {conv:.3e}")
        if conv<= patience or conv>10:
            break
    return np.array(CONV_list)
