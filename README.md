# Natural Language Processing in SpaCy - tutorial

**still work in progress**

A collection of useful functions to create a training dataset on the example of Harry Potter texts.
By default, the off-the-shelf spacy model fails to correctly identify all the character names and label them as person.

- generate_better_characters():
A function that reads in a json file containing a list of the full names of the characters in Harry Potter.
The code then shows how to split those names and generate all possible variations of them that might appear in the text.
