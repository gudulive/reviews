
from utils import train_model
from utils import test_model
from load import load_training_data

if __name__ == "__main__":
    train, test = load_training_data(limit=2500)
    train_model(train, test)
    print("Testing model")
    test_model()# -*- coding: utf-8 -*-

