import json
import numpy as np


def load_data(data_path):

    with open(data_path, "r") as f:
        data = json.load(f)
    
    # convert lists to numpy arrays
    mfcc_to_predict = np.array(data["mfcc"])

    print("Music succesfully loaded!")

    return  mfcc_to_predict

