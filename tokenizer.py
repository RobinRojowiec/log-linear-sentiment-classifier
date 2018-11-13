import re


class Tokenizer:
    def __init__(self, stop_word_file):
        """set of stop word tokens to filter"""
        self.stopwords = set()

        # pattern matches all word characters [a-zA-Z_]
        self.word_pattern = re.compile('^\w+$')

        # loads the stopwords by line from a file
        self.load_stopwords(stop_word_file)

    def create_bow_from_file(self, file_path):
        with open(file_path) as f:
            tokens = f.read().split()

        return self.create_bag_of_words(tokens)

    def create_bag_of_words(self, tokens):
        """Reads the content of a file, tokenize it and collects all types except those filtered
        by the filter method"""
        bag_of_words = set()
        for token in tokens:
            filtered_token = self.filter(token)
            if filtered_token is not None:
                bag_of_words.add(filtered_token)

        return bag_of_words

    def load_stopwords(self, file):
        """loads all stopwords into a set"""
        with open(file) as f:
            for row in f:
                self.stopwords.add(row.replace("\n", ""))

    def filter(self, token):
        """filters all stopwords and characters that are not relevant and returns the token"""
        if token not in self.stopwords and self.word_pattern.match(token):
            return token

        else:
            return None
