import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from articles import articles

import nltk, re
from nltk.util import ngrams
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = stopwords.words('english')
normalizer = WordNetLemmatizer()


#print(articles[0])

def preprocess_text(text):
  cleaned = re.sub(r'\W+', ' ', text).lower()
  tokenized = word_tokenize(cleaned)
  normalized = " ".join([normalizer.lemmatize(token) for token in tokenized if not re.match(r'\d+',token)])
  return normalized


cleaned_articles = [preprocess_text(article) for article in articles]
#print(cleaned_articles[0])


#Getting tf-idf scores to see subjects of the text
vectorizer = TfidfVectorizer(norm=None)
tfidf_scores = vectorizer.fit_transform(cleaned_articles)

feature_names = vectorizer.get_feature_names()
article_index = [f"Article {i+1}" for i in range(len(articles))]

df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
print(df_tf_idf)

for i in range(1,10):
  print(df_tf_idf[[f'Article {i}']].idxmax())




#lets look at the n-grams 
def get_ngrams(num):
    for index,article in enumerate(cleaned_articles):
        tokenized = word_tokenize(article)
        n_gram = list(ngrams(tokenized,int(num)))
        BigramFreq = collections.Counter(n_gram)
        BigramFreq = BigramFreq.most_common(2)

        print(f"Article {index+1} " + str(BigramFreq))


    return None



#generate word cloud for the articles
def generate_wordlcoud(art_num):
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(cleaned_articles[int(art_num)+1])
 
    # plot the WordCloud image                      
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
 
    plt.show()

    return None


num = input("Enter the number of n-grams you would like: ")
get_ngrams(num)

stopwords = set(STOPWORDS)
art_num = input("What article do you want to see a wordcloud for (number of article): ")
generate_wordlcoud(art_num)