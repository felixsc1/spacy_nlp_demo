from spacy.tokens import DocBin
import spacy
import json
import warnings


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def test_model(model, text):
    """
    What spaCy expects as training data:
    list of tuples, each consisting of text and dict, the dict consists of:
    key "entities" and another list of tuples, each having 
    the start and end of the char index and their corresponding label
    TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]
    """
    doc = model(text)
    results = []
    entities = []
    for ent in doc.ents:
        # creating the entities tuple:
        entities.append([ent.start_char, ent.end_char, ent.label_])
    if len(entities) > 0:
        results = [text, {"entities": entities}]
        return results


def read_and_analyze_text(textfile):
    """
    output will be every hp character grouped by chapter as json (used in spaCy v2)
    Some patterns that might still be missed are name abbreviations, e.g. Ron Weasley vs. Ronald Weasley
    Also model can't account for texts with bad OCR, typos, fan fiction that has new names...
    --> to generalize to unseen texts, ML is needed.
    """
    nlp = spacy.load("hp_ner")
    TRAIN_DATA = []
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
                if results != None:
                    TRAIN_DATA.append(results)
    save_data("data/hp_output_training_data.json", TRAIN_DATA)
    return TRAIN_DATA


def convert_to_binary(training_data, output_file):
    """
    For spaCy v3 training data needs to be converted into binary format.
    (smaller file size, faster)
    Code adapted from spacy docs and http://ner.pythonhumanities.com/03_02_train_spacy_ner_model.html
    """
    nlp = spacy.blank("en")

    # the DocBin will store the example documents
    db = DocBin()
    for text, annotations in training_data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            # In case there are problems in text parts, skip them.
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_file)


training_data = read_and_analyze_text("data/hp.txt")
convert_to_binary(training_data, "data/hp_output_training_data.spacy")
