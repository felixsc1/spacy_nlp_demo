# Natural Language Processing in SpaCy - tutorial

**still work in progress**

A collection of useful functions to create a spaCy v3 training dataset on the example of Harry Potter texts.
By default, the off-the-shelf spacy model fails to correctly identify all the character names and label them as person.

The code is mostly based on this [tutorial series](https://www.youtube.com/playlist?list=PL2VXyKi-KpYvuOdPwXR-FZfmZ0hjoNSUo) modified for spaCy v3.

### part1:

- create_training_data():
  A function (and subfunctions) that read in a json file containing a list of the full names of the characters in Harry Potter.
  The code then shows how to split those names and generate all possible variations of them that might appear in the text.

- generate_rules() adds an entity ruler to a blank spacy model using the previously generated patterns.
- read_and_analyze_text() shows an example of how to read and clean up a text, apply the nlp model and store the result

Up to this point, only rules-based approach is used, which has several drawbacks:
The model can't account for texts with bad OCR, typos, or fan fiction that has new names...
To generalize the model to unseen texts, a machine learning approach is needed.

### part2:

Uses entity ruler created in part one to create a training data set, consisting of text and labeled entities in it.
Output is stored as json for review, and converted to .spacy binary format for training in part 3.
