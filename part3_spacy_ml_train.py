import spacy

"""
The training part in spacy v3 is done through the command line.

1. Use the widget in the spacy docs (https://spacy.io/usage/training) to create a base_config.cfg

2. Create the config.cfg with: python -m spacy init fill-config data/base_config.cfg data/config.cfg

3. Start the training process with: python -m spacy train data/config.cfg --output ./models/output
"""

# A models folder is created. To load the best model
trained_nlp = spacy.load("models/output/model-best")
