import json
postings = {}
vocab = []
docids = []


def main():
    read_index_files()

    term_freq = []

    for key, value in postings.items():
        term_val = 0
        for post in value:
            term_val += post[1]
        term_freq.append([vocab[int(key)], term_val])
    term_freq = sorted(term_freq, key=lambda x: x[1], reverse = True)
    print(term_freq[:11])


def read_index_files():
    global postings
    global docids
    global vocab
    # open the files
    in_p = open('postings.txt', 'r')
    in_v = open('vocab.txt', 'r')
    in_d = open('docids.txt', 'r')
    postings = json.load(in_p)
    vocab = json.load(in_v)
    docids = json.load(in_d)
    in_p.close()
    in_d.close()
    in_v.close()
    return

# Standard boilerplate to call the main() function
if __name__ == '__main__':
    main()
