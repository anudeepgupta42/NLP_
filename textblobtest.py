# textblob tutorial

# importing library
from textblob import TextBlob

# creating an instance of Textblob by giving the data
blob = TextBlob("Infosys is an IT company. \n It provides IT support, consulting and recently started its AI domain")

# turning into sentences
blob.sentences

# extracting first sentence
blob.sentences[0]

# printing words of first sentence
for word in blob.sentences[0].words:
    print(word)

# Noun Phrase extraction
blob = TextBlob("Infosys ILP data science specialization course is a great platform to learn Data Science")
for np in blob.noun_phrases:
 print (np)
 
# part of speech tagging
for words, tags in blob.tags:
    print(words, tags)

# word inflection and Lemmatization
blob.words[3].pluralize()  # plural
from textblob import Word
w = Word("Heroes")
w.singularize()  # making a plural word singular

# N- grams
for ngram in blob.ngrams(2):
    print(ngram)
    
# Sentiment analysis
print(blob)
blob.sentiment

# Spelling correction
blob = TextBlob("Ths text has incorrct worts")
blob.correct()
blob.words[4].spellcheck()

# summary 
text = "Analytics Vidhya is a thriving community for data driven industry. This platform allows \
people to know more about analytics from its articles, Q&A forum, and learning paths. Also, we help \
professionals & amateurs to sharpen their skillsets by providing a platform to participate in Hackathons."

from summary import Summary
sm = Summary()
print("this article is about", ','.join(sm.summary(text)))

# translation
blob = TextBlob("This is cool")
blob.translate(from_lang = 'en', to = 'ar')

# Text classification using TextBlob
training = [
('Tom Holland is a terrible spiderman.','pos'),
('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','pos'),
('The Dark Knight Rises is the greatest superhero movie ever!','neg'),
('Fantastic Four should have never been made.','pos'),
('Wes Anderson is my favorite director!','neg'),
('Captain America 2 is pretty awesome.','neg'),
('Let\s pretend "Batman and Robin" never happened..','pos'),
]
testing = [
('Superman was never an interesting character.','pos'),
('Fantastic Mr Fox is an awesome film!','neg'),
('Dragonball Evolution is simply terrible!!','pos')
]

from textblob import classifiers
classifier = classifiers.NaiveBayesClassifier(training)

## decision tree classifier
dt_classifier = classifiers.DecisionTreeClassifier(training)

print (classifier.accuracy(testing))
classifier.show_informative_features(3)

blob = TextBlob('the weather is very good!', classifier=classifier)
print (blob.classify())

