#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


import numpy as np
import pandas as pd
import re
import matplotlib.pyplot
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#Libraries for visualisation
import seaborn as sns
import matplotlib.pyplot as plt

#Libraries for formattting and handling text 
import string 
import re

#Library for nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS


#Library for Splitting Dataset
from sklearn.model_selection import train_test_split


#Libraries for NN
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout

from tensorflow.keras.utils import plot_model

#Library for evaluation
from sklearn import metrics
from functools import reduce
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping



import warnings
warnings.filterwarnings("ignore")


# In[4]:


nltk.download('wordnet')
get_ipython().system('unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/ ')


# In[27]:


test_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv')
test_labels = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')


# In[49]:


test_labels = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')


# In[50]:





# In[5]:


data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')


# In[6]:


data.head()


# In[7]:


print(f"data shape {data.shape}\n\n")
print(f"na summary{data.isna().sum()}\n\n")
print(f'data summary{data.describe()}')


# In[8]:


int_cols = data.select_dtypes(include='int').columns
for col in int_cols:
    print(f"{col},values {data[col].value_counts()} \n\n")


# In[9]:


data[data['obscene'] == 1]


# In[10]:


df = data.copy()
df['toxicity'] = (data[int_cols].sum(axis=1) > 0 ).astype(int)


# In[11]:


df = df[['comment_text','toxicity']]
df.rename(columns = {'comment_text':'text'},inplace = True)


# In[12]:


df.toxicity.unique()


# In[13]:


class Clean:
    def __init__(self, data, col_name = 'text'):
        self.data = data
        self.col_name = col_name
        self.clean_data = self.process()
        
    def remove_URL(self, text):
        return re.sub(r"http[s]?://\S+", "", text)
    
    def remove_emoji(self,text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    def remove_HTML(self, text):
        return re.sub(r"<.*?>", "", text)
    
    def remove_punc(self, text):
        return re.sub('[^a-zA-Z]', ' ', text)
    
    def process(self):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        stop_words.update(['was','does','has','n','s','t'])
        funcs = [self.remove_URL, self.remove_emoji, self.remove_HTML,
                self.remove_punc]
        df = pd.DataFrame()
        df = self.data.copy()
        df[self.col_name] = df[self.col_name].apply(lambda x: x.lower())
        df[self.col_name] = df[self.col_name].apply(lambda x: " "
                                                        .join([word for word in word_tokenize(x) if word not in stop_words]))
        index = 0
        for func in funcs:
            df[self.col_name] = df[self.col_name].apply(lambda x: x.lower())
            
            df[self.col_name] = df[self.col_name].apply(lambda x: func(x))    
            
            df[self.col_name] = df[self.col_name].apply(lambda x:
                                                    ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
        df[self.col_name] = df[self.col_name].apply(lambda x: " "
                                                        .join([word for word in word_tokenize(x) if word not in stop_words]))
        
        return df
    def get_clean_data(self):
        return self.clean_data


# In[14]:


clean = Clean(df)


# In[15]:


df =clean.get_clean_data()


# In[16]:


stopwords.words('english')


# In[17]:


data.iloc[16]


# In[18]:


toxic = df[df['toxicity']==1]
non_toxic = df[df['toxicity']==0]


toxic


# In[19]:


for i in range(100):
    print(toxic.iloc[i].text+'\n\n')


# In[20]:


full_data = [toxic,non_toxic]
for d in full_data:
    wordcloud = WordCloud(width = 1400, height = 700,
                      background_color='white').generate(' '.join(d.text.tolist()))
    fig = plt.figure(figsize = (30,10), facecolor='white')
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title('Most frequent words in toxic comments', fontsize=50)
    plt.tight_layout(pad = 0)
    plt.show()


# In[21]:


l=50
max_features=3000
tokenizer=Tokenizer(num_words=max_features,split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X,maxlen=l)
y = df['toxicity']


# In[22]:


X


# In[23]:


embed_dim = 100
lstm_out = 100
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(lstm_out, dropout=0.2, return_sequences=True,recurrent_dropout=0.4))
model.add(Dropout(0.2))
model.add(LSTM(lstm_out,dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
adam = optimizers.Adam(learning_rate=2e-3)
model.compile(loss = 'binary_crossentropy', optimizer=adam ,metrics = ['accuracy'])
print(model.summary())


# In[24]:


plot_model(model, to_file='model.png')


# In[25]:


es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
mc = ModelCheckpoint ('best_model', monitor='val_metric', mode='min', save_best_only=True)


# 

# In[26]:


model.fit(X,y, epochs = 5,validation_split = 0.2 ,callbacks=[es_callback], batch_size=32)


# In[30]:


test_data


# In[37]:


test_data.rename(columns={'comment_text':'text'},inplace = True)


# In[38]:


test_data


# In[39]:


clean = Clean(test_data,'text')


# In[40]:


test_clean = clean.get_clean_data()


# In[41]:


test_clean


# In[43]:


l=50
max_features=3000
tokenizer=Tokenizer(num_words=max_features,split=' ')
tokenizer.fit_on_texts(test_clean['text'].values)
X_test = tokenizer.texts_to_sequences(test_clean['text'].values)
X_test = pad_sequences(X,maxlen=l)


# In[44]:


test_clean['prediction']= model.predict(X)


# In[45]:


test_clean


# In[46]:


test_labels


# In[51]:


submission = test_clean[['id','prediction']]


# In[52]:


submission


# In[53]:


submission.to_csv('submission.csv',index=False)

