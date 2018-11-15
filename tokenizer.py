#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
        with open(file_path, encoding='utf-8', mode='r') as f:
            tokens = f.read().split()

        return self.create_bag_of_words(tokens)

    def create_bag_of_words(self, tokens, lowercase=True, stem=True):
        """Reads the content of a file, tokenize it and collects all types except those filtered
        by the filter method"""
        bag_of_words = []
        for token in tokens:
            filtered_token = self.filter(token, stem, lowercase)
            if filtered_token is not None:
                bag_of_words.append(filtered_token)

        bow_len = len(bag_of_words)
        big_grams = set()
        for i in range(1, bow_len, 2):
            bi_gram = bag_of_words[i - 1] + "_" + bag_of_words[i]
            #bag_of_words.append(bi_gram)
            big_grams.add(bi_gram)

        return bag_of_words

    def load_stopwords(self, file):
        """loads all stopwords into a set"""
        with open(file) as f:
            for row in f:
                self.stopwords.add(row.replace("\n", "").lower())

    def filter(self, token, stem=True, lowercase=True):
        """filters all stopwords and characters that are not relevant and returns the token"""
        if token.lower() not in self.stopwords and self.word_pattern.match(token):
            if stem:
                token = self.stemmer.stem(token)
            if lowercase:
                return token.lower()
            return token

        else:
            return None
