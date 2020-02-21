#!/usr/bin/python3
# 
import sys
import math
import json
import re
import csv
import nltk
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize, Tree
import matplotlib.pyplot as plt

docids = []
postings = {}
vocab = []
doclengths = []
snippets = {}
titles = {}
named_entities = {}


def main():
    # code for testing offline
    is_file = False
    if len(sys.argv) < 2:
        print('usage: ./indexer.py term [term ...] | -f filename')
        print(len(sys.argv))
        sys.exit(1)

    elif len(sys.argv) == 3:
        if re.match('-f', sys.argv[1]):
            filename = sys.argv[2]
            query_list = read_query_file(filename)
            num_queries = len(query_list)
            is_file = True
        else:
            query_terms = sys.argv[1:]
    else:
        query_terms = sys.argv[1:]
        num_queries = 1

    results = []
    results_snippets = []
    answer = []
    cosine_answer = []
    read_index_files()
    if is_file:
        for query in query_list:
            sep = ' '
            joined_query = sep.join(query)
            if joined_query in named_entities:
                query_named_entities = [[joined_query, 1]]
            else:
                query_named_entities = named_entity_recognition(joined_query)

            print("QNE", query_named_entities)
            query = clean_query_input(query)
            answer = cosineScore(query, query_named_entities, 10)
            #answer = retrieve_vector(query, query_named_entities, 10)
            answer = answer[:10]
            print('Query: ', query)
            i = 0
            current = []
            current_snippets = []
            for docid in answer:
                i += 1
                print(i, docids[docid])
                current.append(docids[docid])
                current_snippets.append(snippets.get(str(docid)))
            results.append(current)
            results_snippets.append(current_snippets)
        write_to_csv("TFIDF_TITLE_NE_FINAL_PORTAL", num_queries, results, results_snippets)
    else:
        query_named_entities = named_entity_recognition(query_terms)
        query_terms = clean_query_input(query_terms)
        cosine_answer = cosineScore(query_terms, query_named_entities, 10)
        print('Query: ', query_terms)
        i = 0
        for docid in cosine_answer:
            i += 1
            print("COSINE", i, docids[docid], snippets.get(str(docid)))


def read_query_file(filename):  # code only for testing offline only - not used for a crawl
    query_list = []
    try:
        input_file = open(filename, 'r')
    except IOError as ex:
        print('Cannot open ', filename, '\n Error: ', ex)
    else:
        query_input = input_file.read().splitlines()  # read the input file
        for query in query_input:
            query_list.append(query.split())
        input_file.close()
    print('QUERY FILE INPUT: ', query_list)
    return query_list


def read_index_files():
    # reads existing data from index files: docids, vocab, postings
    # uses JSON to preserve list/dictionary data structures
    # declare refs to global variables
    global docids
    global postings
    global doclengths
    global vocab
    global snippets
    global titles
    global named_entities
    # open the files
    in_d = open('docids.txt', 'r')
    in_v = open('vocab.txt', 'r')
    in_p = open('postings.txt', 'r')
    in_l = open('doclengths.txt', 'r')
    in_s = open('snippets.txt', 'r')
    in_t = open('titles.txt', 'r')
    in_n = open('named_entities.txt', 'r')
    # load the data
    docids = json.load(in_d)
    vocab = json.load(in_v)
    postings = json.load(in_p)
    doclengths = json.load(in_l)
    snippets = json.load(in_s)
    titles = json.load(in_t)
    named_entities = json.load(in_n)
    # close the files
    in_d.close()
    in_v.close()
    in_p.close()
    in_l.close()
    in_s.close()
    in_t.close()
    in_n.close()
    return


def write_to_csv(system, num_queries, result_list, results_snippets):
    filename = "ranked_output_" + system + ".csv"
    headers = [['Student No.', '100228021'],
               ['System', system],
               ['Query No', 'Rank', 'URL', 'Snippet']]

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(headers)
        for i in range(num_queries):
            for j in range(10):
                # both +1  to change from 0idx to 1idxing
                try:
                    csvwriter.writerow([i + 1, j + 1, result_list[i][j], results_snippets[i][j]])
                except:
                    print("NO MORE ENTRIES")
    return


def retrieve_vector(query_terms, query_named_entities, K):
    global docids  # list of doc names - the index is the docid (i.e. 0-4)
    global doclengths  # number of terms in each document
    global vocab  # list of terms found (237) - the index is the termid
    global postings
    answer = []
    idf = {}
    tfidfs = {}
    query_vector = []

    query_set = set(query_terms)

    idf = inverse_document_frequency(query_terms)

    i = -1
    # For termid in idf dict
    for termid in sorted(idf, key=idf.get, reverse=True):
        # starting i at 0
        i += 1
        # QUERY VECTOR IS NOT NEEDED
        query_vector.append(idf[termid] / len(query_set))
        for post in postings.get(str(termid)):
            if post[0] in tfidfs:
                tfidfs[post[0]] += (idf.get(termid) * post[1]) / doclengths[post[0]] * query_vector[i]
            else:
                tfidfs[post[0]] = (idf.get(termid) * post[1]) / doclengths[post[0]] * query_vector[i]

    #MODIFYING THE SCORE
    tfidfs = named_entity_score_modifier(query_named_entities, tfidfs)
    # NORMALIZING THE LENGTH AND MULITPLYING BY THE TITLE MULTIPLIER
    for docid in tfidfs:
        title_multi = 1
        title_multi = title_multiplier(query_terms, docid)
        tfidfs[docid] *= title_multi
        tfidfs[docid] /= doclengths[docid]

    # for each document in the scores
    # printing them out in descending order (ie highest scoring doc is returned first (ie most relevant doc)
    for docid in sorted(tfidfs, key=tfidfs.get, reverse=True):
        answer.append(docid)

    return answer[:K]


def inverse_document_frequency(query_terms):
    idf = {}
    query_set = set(query_terms)

    for term in query_set:
        try:
            termid = vocab.index(term.lower())
        except:
            print('Not found: ', term, 'is not in vocabulary')
            continue
        idf[termid] = math.log((len(doclengths) / (1 + len(postings.get(str(termid))))))
    return idf


def cosineScore(query_terms, query_named_entities, K):
    global doclengths
    global postings
    global titles
    scores = {}
    query_freq = {}
    answer = []
    # Creating a count for each of the words in the query
    for term in query_terms:
        try:
            termid = vocab.index(term.lower())
        except:
            continue
        if termid in query_freq:
            query_freq[termid] += 1
        else:
            query_freq[termid] = 1

    # Creating the query tfidf vector
    i = -1
    for term in query_terms:
        i += 1
        try:
            termid = vocab.index(term.lower())
        except:
            continue
        for post in postings.get(str(termid)):
            if post[0] in scores:
                scores[post[0]] += (post[1] * (1 + math.log(post[1])))
            else:
                scores[post[0]] = (post[1] * (1 + math.log(post[1])))
    #MODIFYING THE SCORE
    scores = named_entity_score_modifier(query_named_entities,scores)
    # NORMALIZING THE LENGTH AND MULITPLYING BY THE TITLE MULTIPLIER
    for docid in scores:
        title_multi = 1
        title_multi = title_multiplier(query_terms, docid)
        scores[docid] *= title_multi
        scores[docid] /= doclengths[docid]

    for docid in sorted(scores, key=scores.get, reverse=True):
        answer.append(docid)

    return answer[:K]


def named_entity_score_modifier(query_named_entities, scores):
    for ne_list in query_named_entities:
        if ne_list[0] in named_entities:
            ne_multiply_list = named_entities[ne_list[0]]
            for ne_post in ne_multiply_list:
                if ne_post[0] in scores:
                    scores[ne_post[0]] *= 1 + (0.5 * ne_post[1])
    return scores


def clean_query_input(query_terms):
    query_terms = [token.lower() for token in query_terms]
    cleaned = remove_stopwords(query_terms)
    # Can insert lemmatizer here
    cleaned = porter_stemmer(cleaned)
    return cleaned


def title_multiplier(query_terms, docid):
    multiplier = 1
    title = titles[str(docid)]
    title = title.split()
    title = clean_query_input(title)
    for term in query_terms:
        if term.lower() in title:
            multiplier += 0.3
    # print("TITLE: ", title, "MULTIPLICATION: ", multiplier)
    return multiplier


# ALL 3 FUNCTIONS REQUIRE TOKENS TO BE PASSED
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered = []
    for word in tokens:
        if word not in stop_words:
            filtered.append(word)
    return filtered


def porter_stemmer(tokens):
    stemmed = []
    ps = nltk.PorterStemmer()
    for word in tokens:
        stemmed_word = ps.stem(word)
        if word != stemmed_word:
            stemmed.append(stemmed_word)
        else:
            stemmed.append(word)
    return stemmed


def lemmatizer(tokens):
    lemmatized = []
    wordnet = nltk.WordNetLemmatizer()
    for word in tokens:
        lemma_word = wordnet.lemmatize(word)
        if word != lemma_word:
            lemmatized.append(lemma_word)
        else:
            lemmatized.append(word)
    return lemmatized


def named_entity_recognition(query_text):
    chunked = ne_chunk(pos_tag(word_tokenize(query_text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    named_entity_freq = {}
    for chunk in chunked:
        if type(chunk) == Tree:
            current_chunk.append(" ".join([token for token, pos in chunk.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            continuous_chunk.append(named_entity)
            current_chunk = []
        else:
            continue

    # Continuous Chunk now contains a list of all named entities in the text supplied
    for entity in continuous_chunk:
        if entity:
            if entity in named_entity_freq:
                named_entity_freq[entity] += 1
            else:
                named_entity_freq[entity] = 1

    query_named_entities = []
    for key, value in named_entity_freq.items():
        ne_entry = [key, value]
        query_named_entities.append(ne_entry)
    return query_named_entities


# Standard boilerplate to call the main() function
if __name__ == '__main__':
    main()
