import gensim
import nltk
import pandas as pd
import os
import re
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
import numpy
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

os.chdir("C:/Users/sdivy/Downloads")
df = pd.read_csv("upsc.csv", encoding="unicode_escape")
df = df.rename(columns={"col1": "questions", "col2": "answers"})


def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)

    if stopwords:
        sentence = remove_stopwords(sentence)
    return sentence


def get_cleaned_sentences(df, stopwords=False):
    sents = df[["questions"]]
    cleaned_sentences = []

    for index, row in df.iterrows():
        cleaned = clean_sentence(row["questions"], stopwords)
        cleaned_sentences.append(cleaned)
    return cleaned_sentences


cleaned_sentences = get_cleaned_sentences(df, stopwords=True)
cleaned_sentences_with_stopwords = get_cleaned_sentences(df, stopwords=False)

sentences = cleaned_sentences_with_stopwords
sentence_words = [[word for word in document.split()] for document in sentences]

dictionary = corpora.Dictionary(sentence_words)

bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]

v2w_model = None
try:
    v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.mod")
    print("Loaded w2v model")
except:
    v2w_model = api.load("word2vec-google-news-300")
    v2w_model.save("./w2vecmodel.mod")
    print("Saved w2v model")


def getWordVec(word, model):
    samp = model["computer"]
    vec = [0] * len(samp)
    try:
        vec = model[word]
    except:
        vec = [0] * len(samp)
    return vec


def getPhraseEmbedding(phrase, embeddingmodel):
    samp = getWordVec("computer", embeddingmodel)
    vec = numpy.array([0] * len(samp))
    den = 0
    for word in phrase.split():
        den = den + 1
        vec = vec + numpy.array(getWordVec(word, embeddingmodel))
    return vec.reshape(1, -1)


sent_embeddings = []
for sent in cleaned_sentences:
    sent_embeddings.append(getPhraseEmbedding(sent, v2w_model))


def retrieveAndPrintFAQAnswer(
    question_embedding, sentence_embeddings, FAQdf, sentences
):
    max_sim = -1
    index_sim = -1
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        if sim > max_sim:
            max_sim = sim
            index_sim = index

    return FAQdf.iloc[index_sim, 1], max_sim


def get_similar_question_answer(user_question):
    question_orig = user_question
    question = clean_sentence(question_orig, stopwords=False)
    question_embedding = dictionary.doc2bow(question.split())
    question_embedding_w2v = getPhraseEmbedding(question, v2w_model)
    return retrieveAndPrintFAQAnswer(
        question_embedding_w2v, sent_embeddings, df, cleaned_sentences
    )
