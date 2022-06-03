import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
import json


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def generate_better_characters(file):
    """
    json file only complaines the full names of each character.
    Here we create a more comprehensive list, that includes Mr./Mrs. [name], partial names, etc.
    Any instance that could be encountered in texts.
    """
    data = load_data(file)
    new_characters = []
    for item in data:
        new_characters.append(item)
    for item in data:
        # e.g. "Beedle the Bard" -> Beedle Bard
        item = item.replace("The", "").replace(
            "the", "").replace("and", "").replace("And", "")
        names = item.split(" ")
        for name in names:
            name = name.strip()  # remove spaces at beginning/end
            new_characters.append(name)
        # to add "Mad-Eye" from "Alastor (Mad-Eye) Moody":
        if "(" in item:
            names = item.split("(")
            for name in names:
                name = name.replace(")", "").strip()
                new_characters.append(name)
        # to add "Antioch, Cadmus, and Ignotus Peverell" separately
        if "," in item:
            names = item.split(",")
            for name in names:
                name = name.replace("and", "").strip()
                # "Ignotius Peverell" is still not separated, so:
                if " " in name:
                    names2 = name.split()
                    for x in names2:
                        x = x.strip()
                        new_characters.append(x)
                # also add the two word name:
                new_characters.append(name)
    # Next we also add titles, e.g. Uncle/Professor/Mr... to the names
    # Most probably don't exist in the text but thats fine.
    final_characters = []
    titles = ["Dr.", "Professor", "Mr.", "Mrs.",
              "Ms.", "Miss", "Aunt", "Uncle", "Mr. and Mrs."]
    for character in new_characters:
        if "" != character:  # remove any blanks
            final_characters.append(character)
            for title in titles:
                titled_char = f"{title} {character}"
                final_characters.append(titled_char)
    print(len(final_characters))

    # lastly, remove all duplicates by converting to set and back to list
    final_characters = list(set(final_characters))
    print(len(final_characters))

# generate_better_characters("data/hp_characters.json")


def create_training_data():
