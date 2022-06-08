import spacy
from spacy.pipeline import EntityRuler
import json


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)  # indent makes it human readable


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
    # print(len(final_characters))

    # lastly, remove all duplicates by converting to set and back to list
    final_characters = list(set(final_characters))
    # print(len(final_characters))
    final_characters.sort()  # alphabetically
    return final_characters


def create_training_data(file, type):
    # Arranges the list of characters into the patterns format that can be passed into spacy.
    data = generate_better_characters(file)
    patterns = []
    for item in data:
        # spacy expects dict containing at least a label and a pattern for each item
        pattern = {
            "label": type,
            "pattern": item
        }
        patterns.append(pattern)
    return patterns


def generate_rules(patterns):
    """
    Adds the input patterns to a blank nlp model. 
    Note: This process was very different in spacyV2 (older tutorials don't work)
    """
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)

    nlp.to_disk("hp_ner")  # store the model


def test_model(model, text):
    doc = model(text)
    results = []
    for ent in doc.ents:
        results.append(ent.text)
    return results


# patterns = create_training_data("data/hp_characters.json", "PERSON")
# generate_rules(patterns)

def read_and_analyze_text(textfile):
    """
    output will be every hp character grouped by chapter.
    Some patterns that might still be missed are name abbreviations, e.g. Ron Weasley vs. Ronald Weasley
    Also model can't account for texts with bad OCR, typos, fan fiction that has new names...
    --> to generalize to unseen texts, ML is needed.
    """
    nlp = spacy.load("hp_ner")
    ie_data = {}
    with open(textfile, "r") as f:
        text = f.read()
        chapters = text.split("CHAPTER")[1:]
        for chapter in chapters:
            # inspecting hp.txt shows chapter number and title are always preceded by a double linebreak
            chapter_num, chapter_title = chapter.split("\n\n")[0:2]
            chapter_num = chapter_num.strip()
            chapter_title = chapter_title.strip()
            segments = chapter.split("\n\n")[2:]

            hits = []
            for segment in segments:
                segment = segment.strip()
                # linebreaks throw off NLP!
                segment = segment.replace("\n", " ")
                results = test_model(nlp, segment)
                for result in results:
                    hits.append(result)
            ie_data[chapter_num] = hits

    save_data("data/hp_data_output.json", ie_data)
    return ie_data


# read_and_analyze_text("data/hp.txt")
