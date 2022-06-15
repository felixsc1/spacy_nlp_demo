from nltk.corpus import stopwords
import spacy
import string

# execute the first time:
# import nltk
# nltk.download('stopwords')
# and you might need to run:
# python -m spacy download en_core_web_sm

# very short example corpus.
corpus = "Tom is cat, while Jerry is a mouse. Tom and Jerry are characters in a cartoon series. Some of the cartoons contain words, but most are silent. Silent cartoons still have music and sound effects."


# -- PREPROCESSING STEPS -----

stops = set(stopwords.words('english'))
# print(stops)


def remove_stopwords(corpus, stops):
    # Because these throw off ML models.
    corpus = corpus.lower()
    words = corpus.split()

    new_corpus = []
    for word in words:
        if word not in stops:
            new_corpus.append(word)

    corpus = " ".join(new_corpus)
    return corpus


def clean_sentences(corpus):
    """
    Next steps:
    1. use sentence tokenizer to split text into sentences
    2. remove punctuation
    3. make all words lowercase
    4. Optional: lemmatize words to produce smaller amount of vectors.
    5. Split sentence into words and append them to one list per sentence.

    Output is a list of lists as required for gensim.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(corpus)

    sentences = []
    for sent in doc.sents:
        # see https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
        sentence = sent.text.translate(
            str.maketrans('', '', string.punctuation))
        words = sentence.split()
        sentences.append(words)
    return sentences


# -- GENSIM STEPS -----

def create_wordvecs(corpus, model_name):
    from gensim.models.word2vec import Word2Vec

    print("Training model now...")
    w2v_model = Word2Vec(min_count=1,
                         window=2,
                         vector_size=10,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20)

    w2v_model.build_vocab(corpus, progress_per=10000)

    w2v_model.train(
        corpus, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    w2v_model.wv.save_word2vec_format(f"data/{model_name}.txt")


# Loading the gensim word vectors into spacy model

def load_word_vectors(model_name, word_vectors):
    """
    input is the word_vecs.txt created above.
    Function shows how to implement spacy command line code in a script
    using subprocess.
    """
    import subprocess
    import sys
    # print(sys.executable)
    subprocess.run([sys.executable,
                    "-m",
                    "spacy",
                    "init",
                    "vectors",
                    "en",
                    word_vectors,
                    model_name
                    ]
                   )
    print(f"New spaCy model created with word vectors. File: {model_name}")


# --- MAIN - Function execution

corpus_a = remove_stopwords(corpus, stops)
corpus_b = clean_sentences(corpus_a)
# print(corpus3)

create_wordvecs(corpus_b, "word_vecs")

load_word_vectors("data/sample_model", "data/word_vecs.txt")


# -- Next steps:

"""
the spacy model above doesn't contain any pipes.

You then have to load it, add an ner pipe and train it with the training data set.
Explained in this video: https://www.youtube.com/watch?v=JmLQedi80_Y

"""
