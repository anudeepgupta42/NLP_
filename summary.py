from textblob import TextBlob
from textblob import Word

class Summary:
    def __init__(self):
        pass
    def summary(self, text):
        blob = TextBlob(text)
        nouns = set()
        for word,tag in blob.tags:
            if tag == 'NN':
                nouns.add(word)
        words = []
        for item in nouns:
            word = Word(item)
            words.append(word.pluralize())
        return words 
