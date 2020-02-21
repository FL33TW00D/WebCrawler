#!/usr/bin/python3
# 
import sys
import math

docids = ["./LookingGlass/lg001.html/", "./LookingGlass/lg002.html/", "./LookingGlass/lg003.html/", "./LookingGlass/lg004.html/", "./LookingGlass/lg005.html/"]
doclength = {0: 66, 1: 80, 2: 101, 3: 152, 4: 122}
vocab = ["1", "looking", "glass", "house", "one", "thing", "was", "certain", "that", "the", "white", "kitten", "had", "nothing", "to", "do", "with", "it", "black", "fault", "entirely", "for", "been", "having", "its", "face", "washed", "by", "old", "cat", "last", "quarter", "of", "an", "hour", "and", "bearing", "pretty", "well", "considering", "so", "you", "see", "couldn't", "have", "any", "hand", "in", "mischief", "2", "way", "dinah", "her", "children", "faces", "this", "first", "she", "held", "poor", "down", "ear", "paw", "then", "other", "rubbed", "all", "over", "wrong", "beginning", "at", "nose", "just", "now", "as", "i", "said", "hard", "work", "on", "which", "lying", "quite", "still", "trying", "purrno", "doubt", "feeling", "meant", "good", "3", "but", "finished", "earlier", "afternoon", "while", "alice", "sitting", "curled", "up", "a", "corner", "great", "arm", "chair", "half", "talking", "herself", "asleep", "grand", "game", "romps", "ball", "worsted", "wind", "rolling", "till", "come", "undone", "again", "there", "spread", "hearth", "rug", "knots", "tangles", "running", "after", "own", "tail", "middle", "4", "oh", "wicked", "little", "thing'", "cried", "catching", "giving", "kiss", "make", "understand", "disgrace", "really", "ought", "taught", "better", "manners", "know", "ought'", "added", "reproachfully", "speaking", "cross", "voice", "could", "manage", "scrambled", "back", "into", "taking", "began", "winding", "\u0001", "not", "get", "very", "fast", "time", "sometimes", "kitty", "sat", "demurely", "knee", "pretending", "watch", "progress", "putting", "out", "gently", "touching", "if", "would", "be", "glad", "help", "might", "5", "do", "what", "morrow", "is", "kitty?'", "you'd", "guessed", "you'd", "window", "meonly", "making", "tidy", "watching", "boys", "getting", "sticks", "bonfireand", "wants", "plenty", "only", "got", "cold", "snowed", "they", "leave", "off", "never", "mind", "we'll", "go", "bonfire", "morrow'", "here", "wound", "two", "or", "three", "turns", "round", "neck", "how", "look", "led", "scramble", "rolled", "upon", "floor", "yards", "unwound"]
postings = {0: [[0, 2], [1, 1], [2, 1], [3, 1], [4, 1]], 1: [[0, 1], [1, 1], [2, 1], [3, 2], [4, 1]], 2: [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], 3: [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]], 4: [[0, 1], [1, 1], [3, 1]], 5: [[0, 1], [1, 1]], 6: [[0, 2], [1, 4], [2, 2], [3, 2], [4, 2]], 7: [[0, 1]], 8: [[0, 2], [1, 1], [3, 1]], 9: [[0, 6], [1, 6], [2, 8], [3, 11], [4, 8]], 10: [[0, 2], [1, 1]], 11: [[0, 3], [1, 1], [2, 3], [3, 3], [4, 1]], 12: [[0, 4], [2, 5], [4, 1]], 13: [[0, 1]], 14: [[0, 1], [1, 1], [2, 2], [3, 6], [4, 5]], 15: [[0, 1]], 16: [[0, 1], [1, 2], [2, 3], [3, 1], [4, 1]], 17: [[0, 3], [1, 1], [2, 3], [3, 5], [4, 5]], 18: [[0, 1], [2, 1]], 19: [[0, 1]], 20: [[0, 1]], 21: [[0, 2], [1, 1], [4, 1]], 22: [[0, 1], [2, 4], [4, 1]], 23: [[0, 1], [2, 1]], 24: [[0, 1], [1, 3], [2, 1]], 25: [[0, 1], [1, 1]], 26: [[0, 1], [1, 1]], 27: [[0, 1], [1, 1]], 28: [[0, 1], [3, 1]], 29: [[0, 1], [3, 1]], 30: [[0, 1]], 31: [[0, 1]], 32: [[0, 1], [2, 3], [3, 1], [4, 3]], 33: [[0, 1]], 34: [[0, 1]], 35: [[0, 1], [1, 3], [2, 6], [3, 9], [4, 4]], 36: [[0, 1]], 37: [[0, 1]], 38: [[0, 1]], 39: [[0, 1]], 40: [[0, 1], [2, 1], [4, 3]], 41: [[0, 1], [3, 5], [4, 3]], 42: [[0, 1], [4, 2]], 43: [[0, 1]], 44: [[0, 1], [3, 1], [4, 1]], 45: [[0, 1]], 46: [[0, 1]], 47: [[0, 1], [2, 3], [3, 2], [4, 3]], 48: [[0, 1]], 49: [[1, 1]], 50: [[1, 2]], 51: [[1, 1], [3, 2], [4, 1]], 52: [[1, 1], [3, 2]], 53: [[1, 1]], 54: [[1, 1]], 55: [[1, 1], [4, 1]], 56: [[1, 1]], 57: [[1, 3], [3, 5]], 58: [[1, 1]], 59: [[1, 1]], 60: [[1, 1], [2, 1], [4, 1]], 61: [[1, 1]], 62: [[1, 2], [3, 1]], 63: [[1, 1], [3, 2]], 64: [[1, 1]], 65: [[1, 1]], 66: [[1, 2], [2, 2], [3, 1]], 67: [[1, 1], [2, 1]], 68: [[1, 1]], 69: [[1, 1]], 70: [[1, 2], [3, 1]], 71: [[1, 1]], 72: [[1, 1], [4, 1]], 73: [[1, 1], [3, 1]], 74: [[1, 1], [3, 4]], 75: [[1, 1], [4, 1]], 76: [[1, 1]], 77: [[1, 1]], 78: [[1, 1]], 79: [[1, 1], [3, 2]], 80: [[1, 1], [4, 1]], 81: [[1, 1]], 82: [[1, 1]], 83: [[1, 1]], 84: [[1, 1], [2, 1]], 85: [[1, 1]], 86: [[1, 1]], 87: [[1, 1]], 88: [[1, 1]], 89: [[1, 1]], 90: [[2, 1]], 91: [[2, 1], [3, 1]], 92: [[2, 1]], 93: [[2, 1]], 94: [[2, 1]], 95: [[2, 1]], 96: [[2, 2], [3, 1], [4, 2]], 97: [[2, 1]], 98: [[2, 1]], 99: [[2, 3], [3, 2], [4, 1]], 100: [[2, 2], [3, 2], [4, 1]], 101: [[2, 1]], 102: [[2, 1]], 103: [[2, 1], [3, 1]], 104: [[2, 1], [3, 1]], 105: [[2, 2]], 106: [[2, 1], [3, 1]], 107: [[2, 1], [3, 1]], 108: [[2, 1]], 109: [[2, 1]], 110: [[2, 1]], 111: [[2, 1]], 112: [[2, 1], [3, 2], [4, 1]], 113: [[2, 1], [3, 1], [4, 1]], 114: [[2, 1]], 115: [[2, 1]], 116: [[2, 1]], 117: [[2, 1]], 118: [[2, 1]], 119: [[2, 1], [3, 1], [4, 1]], 120: [[2, 1]], 121: [[2, 1]], 122: [[2, 1]], 123: [[2, 1]], 124: [[2, 1]], 125: [[2, 1]], 126: [[2, 1]], 127: [[2, 1]], 128: [[2, 1]], 129: [[2, 1]], 130: [[2, 1]], 131: [[3, 1]], 132: [[3, 1]], 133: [[3, 1]], 134: [[3, 2]], 135: [[3, 1]], 136: [[3, 1]], 137: [[3, 1]], 138: [[3, 1]], 139: [[3, 1]], 140: [[3, 1]], 141: [[3, 1]], 142: [[3, 1]], 143: [[3, 1]], 144: [[3, 2]], 145: [[3, 1]], 146: [[3, 1]], 147: [[3, 1]], 148: [[3, 1], [4, 1]], 149: [[3, 1]], 150: [[3, 1]], 151: [[3, 1]], 152: [[3, 1]], 153: [[3, 1]], 154: [[3, 1]], 155: [[3, 1]], 156: [[3, 1]], 157: [[3, 1]], 158: [[3, 1]], 159: [[3, 1]], 160: [[3, 1]], 161: [[3, 1], [4, 1]], 162: [[3, 2]], 163: [[3, 1], [4, 1]], 164: [[3, 1], [4, 1]], 165: [[3, 1]], 166: [[3, 2]], 167: [[3, 1]], 168: [[3, 1]], 169: [[3, 2]], 170: [[3, 1], [4, 2]], 171: [[3, 1]], 172: [[3, 1]], 173: [[3, 1]], 174: [[3, 1]], 175: [[3, 1]], 176: [[3, 1]], 177: [[3, 1]], 178: [[3, 1]], 179: [[3, 1]], 180: [[3, 1]], 181: [[3, 2], [4, 1]], 182: [[3, 1], [4, 1]], 183: [[3, 1]], 184: [[3, 1]], 185: [[3, 1]], 186: [[3, 1]], 187: [[4, 1]], 188: [[4, 1]], 189: [[4, 1]], 190: [[4, 1]], 191: [[4, 1]], 192: [[4, 1]], 193: [[4, 1]], 194: [[4, 1]], 195: [[4, 1]], 196: [[4, 1]], 197: [[4, 1]], 198: [[4, 1]], 199: [[4, 1]], 200: [[4, 1]], 201: [[4, 1]], 202: [[4, 1]], 203: [[4, 2]], 204: [[4, 1]], 205: [[4, 1]], 206: [[4, 1]], 207: [[4, 1]], 208: [[4, 2]], 209: [[4, 1]], 210: [[4, 1]], 211: [[4, 1]], 212: [[4, 1]], 213: [[4, 1]], 214: [[4, 1]], 215: [[4, 1]], 216: [[4, 1]], 217: [[4, 1]], 218: [[4, 1]], 219: [[4, 1]], 220: [[4, 1]], 221: [[4, 1]], 222: [[4, 1]], 223: [[4, 1]], 224: [[4, 1]], 225: [[4, 1]], 226: [[4, 1]], 227: [[4, 1]], 228: [[4, 1]], 229: [[4, 1]], 230: [[4, 1]], 231: [[4, 1]], 232: [[4, 1]], 233: [[4, 1]], 234: [[4, 1]], 235: [[4, 2]], 236: [[4, 1]]}

def main():
	# code for testing offline	
	if len(sys.argv) < 2:
		print ('usage: ./retriever.py term [term ...]')
		sys.exit(1)
	query_terms = sys.argv[1:]

	answer = []
	answer = retrieve_vector(query_terms)

	print ('Query: ', query_terms)
	i = 0
	for docid in answer:
		i += 1
		print (i, docids[docid[0]])


def retrieve_vector(query_terms):
	## a function to perform vector model retrieval with tf*idf weighting
	#
	global docids		# list of doc names - the index is the docid (i.e. 0-4)
	global doclength	# number of terms in each document
	global vocab		# list of terms found (237) - the index is the termid
	global postings		# postings dictionary; the key is a termid
						# the value is a list of postings entries, 
						# each of which is a list containing a docid and frequency
	answer = []
	#### your code starts here ####

	#### your code ends here ####	
	return answer


	# Standard boilerplate to call the main() function
if __name__ == '__main__':
    main()
