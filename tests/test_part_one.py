from part1_spacy_ml_train import generate_rules, create_training_data, read_and_analyze_text
import spacy


def test_store_load_model():
    """
    tests if all the model generation functions run without error and create output model file.
    """
    patterns = create_training_data("data/hp_characters.json", "PERSON")
    generate_rules(patterns)

    # try to load the model, it should only have the entity ruler component
    nlp2 = spacy.load("hp_ner")
    assert nlp2.component_names[0] == "entity_ruler"


def test_read_and_analyze_text():
    output = read_and_analyze_text("data/hp.txt")
    # lets just check for one example, whether output format seems ok:
    assert "Professor McGonagall" in output["ONE"]
