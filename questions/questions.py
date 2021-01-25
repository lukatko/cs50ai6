import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dct = {}
    for filename in os.listdir(directory):
        dct[filename] = open(os.path.join(directory, filename), 'r', encoding="UTF-8").read()
    return dct


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    final = []
    for i in words:
        i = i.lower()
        flag = 0
        if (i in string.punctuation or i in nltk.corpus.stopwords.words("english")):
            flag = 1
        if not flag:
            final.append(i)
    return final


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = {}
    for k, v in documents.items():
        seen = set()
        for i in v:
            if i not in seen:
                words[i] = words.get(i, 0) + 1
                seen.add(i)
    num_of_documents = len(documents.items())
    for k in words.keys():
        words[k] = math.log(num_of_documents / words[k])
    return words
            
    


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    lst = []
    for k in files.keys():
        suma = 0
        for i in query:
            suma += idfs[i] * files[k].count(i)
        lst.append((suma, k))
    lst.sort()
    return [x[1] for x in lst][::-1][:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    lst = []
    for k in sentences.keys():
        suma = 0
        num_of_words_in_query = 0
        for i in query:
            if (i in sentences[k]):
                num_of_words_in_query += 1
                suma += idfs[i]
        lst.append((suma, num_of_words_in_query / len(sentences[k]), k))
    lst.sort()
    return [x[2] for x in lst][::-1][:n]


if __name__ == "__main__":
    main()
