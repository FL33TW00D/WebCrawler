import sys
import re
import json

# global declarations for doclist, postings, vocabulary
docids = []  # Stores the relevant URL information
postings = {}  # Postings holds {wordID: ["docID1", "docID2" ... ]} dict{wordID: list of docs containing word}
vocab = []  # Stores a list of all the words in all of the documents


def main():
    # code for testing offline  
    if len(sys.argv) < 2:
        print('usage: ./retriever.py term [term ...]')
        sys.exit(1)
    query_terms = sys.argv[1:]
    answer = []

    read_index_files()

    answer = retrieve_bool(query_terms)
    print('Query: ', query_terms, '\n')

    i = 0
    # Fetching all matching documents in answer
    for docid in answer:
        i += 1
        # Formatting User Output
        print('Count:', i, ' \t Document Containing query: ', docids[int(docid)])


def read_index_files():
    # reads existing data from index files: docids, vocab, postings
    # uses JSON to preserve list/dictionary data structures
    # declare refs to global variables
    global docids
    global postings
    global vocab
    # open the files
    in_d = open('docids.txt', 'r')
    in_v = open('vocab.txt', 'r')
    in_p = open('postings.txt', 'r')
    # load the data
    docids = json.load(in_d)
    vocab = json.load(in_v)
    postings = json.load(in_p)
    # close the files
    in_d.close()
    in_v.close()
    in_p.close()
    return


def retrieve_bool(query_terms):
    global docids  # list of docs crawled
    global postings  # {termid:{docid:freq}}
    global vocab  # list of unique terms found in terms
    answer = []
    merged = []
    operator = ''
    # Fetching the postings within the post for required vocab
    # Using str as out postings store lists
    try:
        for post in postings.get(str(vocab.index(query_terms[0].lower()))):
            answer.append(post)
    except:
        print('Query Term: ', query_terms[0], ' Not found in vocab')

    for term in query_terms:
        # Populating the operators list
        if term in ('AND', 'OR', 'NOT'):
            operator = term
            continue
        try:
            termid = vocab.index(term.lower())
        except:
            # if term is not in vocab and previous results are AND'd with it
            # then answer must be empty
            if operator == 'AND':
                answer = []
            print('Query Term: ', term, ' Not found in vocab')
            continue
        # if the operator is OR
        # simply append to the current answer we have
        if operator == 'OR':
            for post in postings.get(str(termid)):
                answer.append(post)
            answer = sorted(list(set(answer)))
            operator = ''

        elif operator == 'AND':
            merged = answer[:]
            answer = []
            for post in postings.get(str(termid)):
                # If the post we are inspecting already exists(ie we are checking on the posts from the already
                # parsed terms) then we can add it to the list (ie the post contains both terms)
                if post in merged:
                    answer.append(post)
            answer = sorted(list(set(answer)))
            merged = []
            operator = ''

        elif operator == 'NOT':
            for post in postings.get(str(termid)):
                # if the post already exists and contains the NOTTED term remove it from the answer
                if post in answer:
                    answer.remove(post)
            operator = ''
    return answer


if __name__ == '__main__':
    main()
