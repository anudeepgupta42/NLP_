# Gensim tutorial

#################### word to vector
import pandas as pd
import nltk
import gensim
from gensim import corpora, models, similarities

df = pd.read_csv("jokes.csv")

x = df['Question'].values.tolist()
y = df['Answer'].values.tolist()

corpus = x + y

tok_corp = [nltk.word_tokenize(sent) for sent in corpus]

model = gensim.models.Word2Vec(tok_corp, min_count = 1, size = 32)

model.wv.most_similar("man")
model.wv.__getitem__('man')


# Summary

from gensim.summarization import summarize

sentence="Automatic summarization is the process of shortening a text document with software, in order to create a summary with the major points of the original document. Technologies that can make a coherent summary take into account variables such as length, writing style and syntax.Automatic data summarization is part of machine learning and data mining. The main idea of summarization is to find a subset of data which contains the information of the entire set. Such techniques are widely used in industry today. Search engines are an example; others include summarization of documents, image collections and videos. Document summarization tries to create a representative summary or abstract of the entire document, by finding the most informative sentences, while in image summarization the system finds the most representative and important (i.e. salient) images. For surveillance videos, one might want to extract the important events from the uneventful context.There are two general approaches to automatic summarization: extraction and abstraction. Extractive methods work by selecting a subset of existing words, phrases, or sentences in the original text to form the summary. In contrast, abstractive methods build an internal semantic representation and then use natural language generation techniques to create a summary that is closer to what a human might express. Such a summary might include verbal innovations. Research to date has focused primarily on extractive methods, which are appropriate for image collection summarization and video summarization."
summarize(sentence)

sentence ="Holi is one of the famous festivals celebrated all over India and in other countries of South Asia with great enthusiasm. Holi is a festival of colors and people spray colors on one another on this day. Holi is a festival of happiness which signifies the arrival of the spring season; the celebrations are arranged as a thanksgiving for a good harvest. Holi is considered as the mark of hope and joy. The festival also signifies the victory of good over evil. The festival generally falls in the month of March and sometimes in the month of February. Holi falls on March 23 for the year 2016."
summarize(sentence)


sentence = "In 1981, seven engineers started Infosys Limited with just US$250. From the beginning, the company was founded on the principle of building and implementing great ideas that drive progress for clients and enhance lives through enterprise solutions. For over three decades, we have been a company focused on bringing to life great ideas and enterprise solutions that drive progress for our clients. \
We recognize the importance of nurturing relationships that reflect our culture of unwavering ethics and mutual respect. It’ll come as no surprise, then, that 98.3 percent (Q3 FY 18) of our revenues come from existing clients. \
Infosys has a growing global presence with more than 200,000+ employees. Globally, we have 84 sales and marketing offices and 116 development centers as on March 31, 2017. \
At Infosys, we believe our responsibilities extend beyond business. That is why we established the Infosys Foundation – to provide assistance to some of the more socially and economically depressed sectors of the communities in which we work. And that is why we behave ethically and honestly in all our interactions – with our clients, our partners and our employees."


text = "Thomas A. Anderson is a man living two lives. By day he is an " + \
    "average computer programmer and by night a hacker known as " + \
    "Neo. Neo has always questioned his reality, but the truth is " + \
    "far beyond his imagination. Neo finds himself targeted by the " + \
    "police when he is contacted by Morpheus, a legendary computer " + \
    "hacker branded a terrorist by the government. Morpheus awakens " + \
    "Neo to the real world, a ravaged wasteland where most of " + \
    "humanity have been captured by a race of machines that live " + \
    "off of the humans' body heat and electrochemical energy and " + \
    "who imprison their minds within an artificial reality known as " + \
    "the Matrix. As a rebel against the machines, Neo must return to " + \
    "the Matrix and confront the agents: super-powerful computer " + \
    "programs devoted to snuffing out Neo and the entire human " + \
    "rebellion. "
    
text = "Dear Infoscion," + \
"Today is Women’s Day – a day when the world calls for gender equality in unison. Here at Infosys, we are committed to doing our part, in big ways and small. We strive to provide a workplace that is focused on equality in all forms, and at all points in an Infoscion’s career with us. What this means is that, among other things, all of us can go about our work knowing fully well that a job well done will bring rewards that are no different for women and men." + \
"Our Global Diversity Council, led by Aruna Newton, Narsimha Rao (Narry), Karmesh Vaswani, Sangita Singh and Sharmistha Adhya, is working towards empowering women in as many ways as possible. From hiring diversity and inclusion of women in executive leadership to tailored mentoring programs, making it easier for them to rejoin after maternity and creating broader sensitivity around gender-related themes, the Council reports their progress every quarter to the Board." + \
"In fact, I am quite impressed with progress made by the Diversity Council in increasing the number of women technology architects on our projects. This global, business-led effort to reskill and mentor women will see its 450-strong first batch of graduating women technology architects bringing their expertise to work for clients in the next 60 days." + \
"This is important, as our focus on Diversity and Inclusion has a large impact on our business too. While our workforce, powered by AI and automation, can help solve most of our client’s concerns, some problems require more than technology. They require uniquely human thought and action, which is strengthened by diversity in all forms – culture, experience and gender. Here is where, by bringing the need for inclusivity to the fore, we can bring something so uniquely Infosys to our clients, and lead the path to success." + \
"Folks, we’ve come a long way but there’s more to do. Each of us has a role to play in getting closer to our Diversity goals. It takes each of us to make a slight change to make a big impact. It starts with simple steps. Observe your behavior, your mindset and question your assumptions about gender. By learning what you need to do, you’re halfway there. We’re with you, too. "



print(summarize(text, ratio = 0.2))

# keywords extraction
from gensim.summarization import keywords
print(keywords(text))

'''
# Topic modelling
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete] 


# Importing Gensim
import gensim
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)
# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=3))

'''

