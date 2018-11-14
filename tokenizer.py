import re
from nltk.stem.porter import PorterStemmer

class Tokenizer:
    def __init__(self, stop_word_file):
        """set of stop word tokens to filter"""
        self.stopwords = set()

        # pattern matches all word characters [a-zA-Z_]
        self.word_pattern = re.compile('^\w+$')

        # loads the stopwords by line from a file
        self.load_stopwords(stop_word_file)

        self.stemmer = PorterStemmer()

    def create_bow_from_file(self, file_path):
        with open(file_path) as f:
            tokens = f.read().split()

        return self.create_bag_of_words(tokens)

    def create_bag_of_words(self, tokens, lowercase = True, bigrams = True):
        """Reads the content of a file, tokenize it and collects all types except those filtered
        by the filter method"""
        bag_of_words = set()
        for token in tokens:
            filtered_token = self.filter(token, lowercase)
            if filtered_token is not None:
                bag_of_words.add(filtered_token)

        if bigrams:
            combined_bow = set()
            last_word = None
            for word in bag_of_words:
                combined_bow.add(word)

                if last_word is not None:
                    bigram: "" = last_word + "_" + word
                    combined_bow.add(bigram)
                last_word = word

            return combined_bow

        return bag_of_words

    def load_stopwords(self, file):
        """loads all stopwords into a set"""
        with open(file) as f:
            for row in f:
                self.stopwords.add(row.replace("\n", "").lower())

    def filter(self, token, lowercase = False):
        """filters all stopwords and characters that are not relevant and returns the token"""
        token = self.stemmer.stem(token)

        if token.lower() not in self.stopwords and self.word_pattern.match(token):
            if lowercase:
                return token.lower()
            return token

        else:
            return None
