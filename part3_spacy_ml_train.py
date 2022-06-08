import spacy
import json


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    for _, annotations in TRAIN_DATA:
        # reminder: get() returns none, if entry doesn't exist while dictionary["key"] throws error.
        for ent in annotations.get("entities"):
            # here, this just grabs "PERSON", but data set could contain many different labels
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()


TRAIN_DATA = load_data("data/hp_output_training_data.json")
