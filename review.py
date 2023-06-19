import joblib
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ',df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()

    review = [ps.stem(word) for word in review
                if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)

    corpus.append(review)

cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)

model = RandomForestClassifier(n_estimators = 501,
                               criterion='entropy')

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test, y_pred))

joblib.dump(model,"rr_model.sav")




