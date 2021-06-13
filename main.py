import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

trueNews = pd.read_csv("True.csv")
fakeNews = pd.read_csv("Fake.csv")

trueNews["label"] = 1
fakeNews["label"] = 0

trueNews = trueNews[0:5000]
fakeNews = fakeNews[0:5000]

fullSet = [trueNews, fakeNews]
df = pd.concat(fullSet)

stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "%"]
stoplist = text.ENGLISH_STOP_WORDS.union(punc)

# TFIDF for Title
headlines = df["title"].values
headlines = headlines.flatten()
vectorizer = TfidfVectorizer(stop_words=stoplist, tokenizer=tokenize, max_features=250)
titleMatrix = vectorizer.fit_transform(headlines)
textKeyWords = vectorizer.get_feature_names()
titleTFIDF = pd.DataFrame.sparse.from_spmatrix(titleMatrix)


# TFIDF for Text
text = df["text"].values
text = text.flatten()
vectorizer2 = TfidfVectorizer(stop_words=stoplist, tokenizer=tokenize, max_features=500)
textMatrix = vectorizer2.fit_transform(text)
textKeywords = vectorizer2.get_feature_names()
textTFIDF = pd.DataFrame.sparse.from_spmatrix(textMatrix)


# KMeans for Title
model = KMeans(n_clusters=10, n_init=20, n_jobs=1)
model.fit(titleMatrix)  # computer kmeans cluster on Rooms, Type, Regionname, Price
x = model.fit_predict(titleMatrix)  # computes the cluster center and predicts cluster index
kMeansTitle = pd.DataFrame(x)


# KMeans for Text
model = KMeans(n_clusters=20, n_init=20, n_jobs=1)
model.fit(textMatrix)  # computer kmeans cluster on Rooms, Type, Regionname, Price
x = model.fit_predict(textMatrix)  # computes the cluster center and predicts cluster index
kmeansText = pd.DataFrame(x)


# VADER Sentiment for Title
analyser = SentimentIntensityAnalyzer()
corpus = list(df['title'])
sentimentscores = []
for i in corpus:
    score = analyser.polarity_scores(i)
    score['title'] = i
    sentimentscores.append(score)
sentimentTitle = pd.DataFrame(sentimentscores)
sentimentTitle.drop(columns=['title'], inplace=True)
sentimentTitle.columns = ["titleNeg", "titleNeu", "titlePos", "titleComp"]


# VADER Sentiment for Text
analyser2 = SentimentIntensityAnalyzer()
corpus = list(df['text'])
sentimentscores = []
for i in corpus:
    score = analyser2.polarity_scores(i)
    score['text'] = i
    sentimentscores.append(score)
sentimentText = pd.DataFrame(sentimentscores)
sentimentText.drop(columns=['text'], inplace=True)
sentimentText.columns = ["textNeg", "textNeu", "textPos", "textComp"]


# Making a singular Data Frame

combine = [titleTFIDF, textTFIDF, kMeansTitle, kmeansText, sentimentTitle, sentimentText]

finalDataFrame = pd.concat(combine, axis=1, sort=False)


label = df["label"]

X_train, X_test, y_train, y_test = train_test_split(finalDataFrame, label, random_state=0)

Adab= AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=5)
model = Adab.fit(X_train, y_train)
score2 = Adab.score(X_test, y_test)
print(score2)


titles_options = ["Confusion Matrix of Fake News Classifier"]
for title in titles_options:
    disp = plot_confusion_matrix(Adab, X_test, y_test,
                                 display_labels=["True", "False"],
                                 cmap=plt.cm.Blues,
                                 )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()