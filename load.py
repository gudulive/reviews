# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:29:04 2020

@author: Joel Sifuya
"""

import os
import random
from utils import train_model
from utils import test_model
from load import load_training_data

# define the directory in which your data is stored 
# the ratio of training data to test data
# A good ratio to start with is 80 percent of the data for training data and 20 percent for test data

def load_training_data(
    data_directory: str = "Data/aclImdb/train",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
# Load from files
# constructing the directory structure of the data, looking for and opening text files, then appending a tuple of the contents and a label dictionary to the reviews list.
   
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label}
                        }
                        reviews.append((text, spacy_label))
                        
# shuffle to eliminate any possible bias from the order in which training data is loaded using random module

    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]

# Modifying the base spaCy pipeline to include the textcat component
# Building a training loop to train the textcat component
# Evaluating the progress of your model training after a given number of training loops

# from utils import train_model
# from utils import test_model
# from load import load_training_data


if __name__ == "__main__":
    train, test = load_training_data(limit=2500)
    train_model(train, test)
    print("Testing model")
    test_model()# -*- coding: utf-8 -*-
