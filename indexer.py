#!/usr/bin/python3
# PCcrawler.py inforet.cmp.uea.ac.uk http://inforet.cmp.uea.ac.uk/

import sys
import re
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize, Tree
from bs4 import BeautifulSoup


docids = []  # Stores the relevant URL information
postings = {}  # Postings holds {wordID: ["docID1", "docID2" ... ]} dict{wordID: list of docs containing word}
vocab = []  # Stores a list of all the words in all of the documents
doclengths = []  # Stores the number of terms in each document
named_entities = {} # SHOULD NOW STORE "NAMED ENTITY" : [DOCID, FREQ]
snippets = {} # MATCHES DOCID(INTEGER), TO SNIPPET OF THAT DOCID
titles = {} # MATCHES DOCID(INTEGER), TO TITLE OF THAT DOCUMENT

def main():
    # code only for testing offline only - not used for a crawl
    max_files = 32000
    if len(sys.argv) == 1:
        print('usage: ./indexer.py file | -d directory [maxfiles]')
        sys.exit(1)
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
    elif len(sys.argv) == 3:
        if re.match('-d', sys.argv[1]):
            dirname = sys.argv[2]
            dir_index = True
        else:
            print('usage: ./indexer.py file | -d directory [maxfiles]')
            sys.exit(1)
    elif len(sys.argv) == 4:
        if re.match('\d+', sys.argv[3]):
            max_files = int(sys.argv[3])
        else:
            print('usage: ./indexer.py file | -d directory [maxfiles]')
            sys.exit(1)
    else:
        print('usage: ./indexer.py file | -d directory [maxfiles]')

    if len(sys.argv) == 2:
        index_file(filename)
    elif re.match('-d', sys.argv[1]):
        for filename in os.listdir(sys.argv[2]):
            if re.match('^_', filename):
                continue
            if max_files > 0:
                max_files -= 1
                filename = sys.argv[2] + '/' + filename
                index_file(filename)
            else:
                break

    write_index_files()


def index_file(filename):  # code only for testing offline only - not used for a crawl
    try:
        input_file = open(filename, 'rb')
    except IOError as ex:
        print('Cannot open ', filename, '\n Error: ', ex)
    else:
        page_contents = input_file.read()  # read the input file
        url = 'http://www.' + filename + '/'
        make_index(url, page_contents)
        input_file.close()


def clean_html(html, docid):
    global named_entities
    global titles
    create_titles(html, docid)
    no_title = re.sub(r'<title>.+<\/title>', '', html)
    soup = BeautifulSoup(no_title, 'html.parser')
    # Using snippet from: https://stackoverflow.com/questions/22799990/beatifulsoup4-get-text-still-has-javascript
    for script in soup(["script", "style"]):
        script.decompose()
    cleaned = soup.get_text()
    # create_snippet(cleaned, docid)
    named_entity_recognition(cleaned, docid)
    # REMOVING ALL PUNCTUATION APART FROM APOSTROPHE
    cleaned = re.sub(r"[^\w\d'\s]+", '', cleaned)
    # REMOVING THE EM-DASH ASCII THAT IS PRESENT THROUGHOUT
    cleaned = re.sub(r'\u2014', ' ', cleaned)
    # REMOVING ALL OF THE LG NAMES SCRAPED FROM HOMEPAGE
    cleaned = re.sub(r"lg\d{0,3}html", '', cleaned)
    # REMOVING INDEX FROM HOMEPAGE AS IT'S NOT WITHIN DESIRED TEXT
    # REMOVING ALL THE DIGITS SCRAPED FROM THE HOMEPAGE
    cleaned = re.sub(r"\d+", '', cleaned)
    # REGEX TO REMOVE APOSTROPHE AT START AND FINISH
    cleaned = re.sub(r"\B'\b|\b'\B", '', cleaned)
    # SETTING ALL TO LOWER CASE
    cleaned = cleaned.lower()
    return cleaned.strip()


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


def named_entity_recognition(text, docid):
    global named_entities
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
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

    for entity in continuous_chunk:
        if entity:
            if entity in named_entity_freq:
                named_entity_freq[entity] += 1
            else:
                named_entity_freq[entity] = 1

    for named_entity in named_entity_freq:
        ne_post = [int(docid), named_entity_freq[named_entity]]
        if named_entity not in named_entities:
            named_entities[named_entity] = [ne_post]
        else:
            named_entities[named_entity].append(ne_post)

    return


def write_index_files():
    global docids
    global postings
    global vocab
    global doclengths
    global snippets
    out_d = open('docids.txt', 'w')
    out_l = open('doclengths.txt', 'w')
    out_v = open('vocab.txt', 'w')
    out_p = open('postings.txt', 'w')
    out_s = open('snippets.txt', 'w')
    out_t = open('titles.txt', 'w')
    out_n = open('named_entities.txt', 'w')
    json.dump(docids, out_d)
    json.dump(doclengths, out_l)
    json.dump(vocab, out_v)
    json.dump(postings, out_p)
    json.dump(snippets, out_s)
    json.dump(titles, out_t)
    json.dump(named_entities, out_n)
    out_d.close()
    out_l.close()
    out_v.close()
    out_p.close()
    out_s.close()
    out_t.close()
    out_n.close()
    d = len(docids)
    v = len(vocab)
    p = len(postings)
    t = len(titles)
    ne = len(named_entities)
    print('===============================================')
    print('Indexing: ', d, ' docs ', v, ' terms ', 'named entities', ne , 'titles', t, 'and', p, ' postings lists written to file')
    print('===============================================')
    print('INDEX CREATED SUCCESSFULLY')
    return


def read_index_files():
    global docids
    global postings
    global vocab
    global doclengths
    in_d = open('docids.txt', 'r')
    in_l = open('doclengths.txt', 'r')
    in_v = open('vocab.txt', 'r')
    in_p = open('postings.txt', 'r')
    docids = json.load(in_d)
    vocab = json.load(in_v)
    doclengths = json.load(in_l)
    postings = json.load(in_p)
    in_d.close()
    in_v.close()
    in_p.close()
    return


def create_snippet(page_text, docid):
    global snippets
    sent = page_text.split()
    seperator = ' '
    snippet = seperator.join(sent[:30]) + '...'
    snippets[docid] = snippet


def create_titles(html, docid):
    global titles
    title_soup = BeautifulSoup(html, 'html.parser')
    titles[int(docid)] = title_soup.find('title').text


def make_index(url, page_contents):
    global docids
    global postings
    global vocab
    global doclengths
    # extract the words from the page contents
    if isinstance(page_contents, bytes):  # convert bytes to string if necessary
        page_contents = page_contents.decode('utf-8', 'ignore')

    #CONSTRUCTING THE DOCIDS DICT
    # REMOVE HTTP(S)
    if re.search('https:..', url):
        domain_url = re.sub('https://', '', url)
    elif re.search('http:..', url):
        domain_url = re.sub('http://', '', url)
    else:
        print("make_index: no match for protocl url=", url)

    # Removing world wide web prefix
    if re.search('www.', domain_url):
        domain_url = re.sub('www.', '', domain_url)
    if domain_url in docids:
        return
    else:
        docids.append(domain_url)
        # docid stores the index of the document in the docids
        docid = str(docids.index(domain_url))
    page_text = clean_html(page_contents, docid)
    # print('=================================================================')
    # print('make_index: url = ', url)
    # print('make_index: page_text= ', page_text)
    #NE = named_entities.get(int(docid))
    current_title = titles.get(int(docid))
    print("DOCUMENT TITLE: ", current_title)
    terms = page_text.split()

    # PERFORM STEMMING AND LEMMATIZING
    terms = remove_stopwords(terms)
    terms = porter_stemmer(terms)
    #terms = lemmatizer(terms)
    # END STEMMING AND LEMMATIZING

    # docfreq is for this one document
    docfreq = {}
    # variable to hold the number of words in this particular doc
    docwordscount = 0

    for word in terms:
        docwordscount += 1
        # IF WORD IS ALREADY IN VOCAB, THEN JUST EXTRACT ID FROM VOCAB LIST AND INCREMENT IN DOCFREQ DICT
        if word in vocab:
            wordid = str(vocab.index(word))
        else:
            vocab.append(word)
            wordid = str(vocab.index(word))

        # Increasing count upon each occurance of word
        if wordid in docfreq:
            docfreq[wordid] += 1
        else:
            docfreq[wordid] = 1

    # Appending the number of words in the doc to the global doclengths
    doclengths.append(docwordscount)

    # to add term frequencies to the postings need docid + count for each term in the document
    for wordid in docfreq:
        docf = [int(docid), docfreq[wordid]]
        if wordid not in postings:
            postings[wordid] = []
            postings[wordid].append(docf)
        else:
            postings[wordid].append(docf)
    return

# Standard boilerplate to call the main() function
if __name__ == '__main__':
    main()
