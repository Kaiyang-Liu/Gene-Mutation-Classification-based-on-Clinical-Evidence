# **FYP Final Version - LightGBM + (Bi-GRU and Attention)**

# # Packages Loading
import re
import time
import os
import math
import pandas as pd
import numpy as np

# Preprocess
import spacy
from scipy.stats import entropy
import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
nltk.download('punkt')
stop_words = set(stopwords.words('english')) 
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Gensim
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
from gensim.models.word2vec import Word2Vec

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import scikitplot.plotters as skplt

# Modeling
import lightgbm as lgb
from datetime import datetime

# Viz
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Kears
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.layers.merge import concatenate
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import adam_v2

from keras import backend as K
import tensorflow.python.keras.engine
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints

# 搭建模型
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, Input
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D
from keras.layers import  BatchNormalization
from keras.layers import Convolution1D, Conv1D,MaxPooling1D
from keras.layers import Dense, Embedding, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional, merge, Input, concatenate
from keras.layers.merge import Concatenate


# # Data Exploration

# ## Loading Data

train_variants_df = pd.read_csv("training_variants.csv")
train_text_df = pd.read_csv("training_text.zip",sep="\|\|",engine="python",names=["ID","Text"],skiprows=1)
test_variants_df = pd.read_csv("stage2_test_variants.csv")
test_text_df = pd.read_csv("stage2_test_text.csv",sep="\|\|",engine="python",names=["ID","Text"],skiprows=1)

val_variants_df = pd.read_csv("test_variants.csv")
val_text_df = pd.read_csv("test_text.zip",sep="\|\|",engine="python",names=["ID","Text"],skiprows=1)
val_labels_df = pd.read_csv("stage1_solution_filtered.csv")
val_labels_df['Class'] = pd.to_numeric(val_labels_df.drop('ID', axis=1).idxmax(axis=1).str[5:])
val_labels_df = val_labels_df[['ID', 'Class']]
val_text_df = pd.merge(val_text_df, val_labels_df, how='left', on='ID')

print("Train Variant".ljust(15), train_variants_df.shape)
print("Train Text".ljust(15), train_text_df.shape)
print("Test Variant".ljust(15), test_variants_df.shape)
print("Test Text".ljust(15), test_text_df.shape)
print("Validation Variant".ljust(15), val_variants_df.shape)
print("Validation Text".ljust(15), val_text_df.shape)

train_variants_df.head()
test_variants_df.head()
val_variants_df.head()
train_variants_df['Class'].value_counts()
train_text_df
test_text_df
val_text_df

# ## Distribution of genetic mutation classes

plt.figure(figsize=(14,8))
sns.countplot(x="Class", data=train_variants_df, palette="PuBu")
plt.ylabel('Frequency', fontsize=22)
plt.xlabel('Classes of Genes Variation', fontsize=22)
plt.title("Distribution of Genetic Mutation Classes", fontsize=24)
plt.show()

gene_group = train_variants_df.groupby("Gene")['Gene'].count()
minimal_occ_genes = gene_group.sort_values(ascending=True)[:10]
print("Genes with maximal occurences\n", gene_group.sort_values(ascending=False)[:10])
print("\nGenes with minimal occurences\n", minimal_occ_genes)


# ## Distribution of Gene in Different Classes

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20,15))

for i in range(3):
    for j in range(3):
        gene_count_grp = train_variants_df[train_variants_df["Class"]==((i*3+j)+1)]
        .groupby('Gene')["ID"].count().reset_index()
        sorted_gene_group = gene_count_grp.sort_values('ID', ascending=False)
        sorted_gene_group_top_7 = sorted_gene_group[:7]
        sns.barplot(x="Gene", y="ID", data=sorted_gene_group_top_7, ax=axs[i][j],palette="PuBu")


# Some points we can conclude from these graphs:
# 
# BRCA1 is highly dominating Class 5\
# SF3B1 is highly dominating Class 9\
# BRCA1 and BRCA2 are dominating Class 6

# # Data Preprocessing

# ## Text Preprocessing

# **Steps**\
# **1. Tokenization**\
# **2. Removal of punctuations**\
# **3. Lemmatization**\
# **4. Removal of stop words**\
# **5. Lower casting**\
# **6. Special consideration for clinical text***

# Remove punctuation
train_text_df['Text_processed'] = train_text_df['Text'].map(lambda x: re.sub('[\',.!?*]', '', str(x)))
# Convert the titles to lowercase
train_text_df['Text_processed'] = train_text_df['Text_processed'].replace(r'\n',' ', regex=True) 
train_text_df['Text_processed'] = train_text_df['Text_processed'].map(lambda x: x.lower())
# Print out the first rows of papers
train_text_df['Text_processed'] = train_text_df['Text_processed'].apply(lambda x: x.strip())
train_text_df

# Remove punctuation
test_text_df['Text_processed'] = test_text_df['Text'].map(lambda x: re.sub('[\',.!?*]', '', str(x)))
# Convert the titles to lowercase
test_text_df['Text_processed'] = test_text_df['Text_processed'].replace(r'\n',' ', regex=True) 
test_text_df['Text_processed'] = test_text_df['Text_processed'].map(lambda x: x.lower())
# Print out the first rows of papers
test_text_df['Text_processed'] = test_text_df['Text_processed'].apply(lambda x: x.strip())
test_text_df

# Remove punctuation
val_text_df['Text_processed'] = val_text_df['Text'].map(lambda x: re.sub('[\',.!?*]', '', str(x)))
# Convert the titles to lowercase
val_text_df['Text_processed'] = val_text_df['Text_processed'].replace(r'\n',' ', regex=True) 
val_text_df['Text_processed'] = val_text_df['Text_processed'].map(lambda x: x.lower())
# Print out the first rows of papers
val_text_df['Text_processed'] = val_text_df['Text_processed'].apply(lambda x: x.strip())
val_text_df

train_full = train_variants_df.merge(train_text_df, how="inner", left_on="ID", right_on="ID")
train_full = train_full.drop("Text", axis=1)
train_full.head()

test_full = test_variants_df.merge(test_text_df, how="inner", left_on="ID", right_on="ID")
test_full = test_full.drop("Text", axis=1)
test_full.head()

val_full = val_variants_df.merge(val_text_df, how="inner", left_on="ID", right_on="ID")
val_full = val_full.drop("Text", axis=1)
val_full.head()

import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())
train_full['Text_processed'] = train_full['Text_processed'].apply(lambda x: " "
    .join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))
test_full['Text_processed'] = test_full['Text_processed'].apply(lambda x: " "
    .join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))
val_full['Text_processed'] = val_full['Text_processed'].apply(lambda x: " "
    .join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words ))

# ## Word Statistics

train_text_df.loc[:, 'Text_count']  = train_text_df["Text_processed"].apply(lambda x: len(x.split()))
train_text_df.head()
test_text_df.loc[:, 'Text_count']  = test_text_df["Text_processed"].apply(lambda x: len(x.split()))
test_text_df.head()
val_text_df.loc[:, 'Text_count']  = val_text_df["Text_processed"].apply(lambda x: len(x.split()))
val_text_df.head()
train_full = train_variants_df.merge(train_text_df, how="inner", left_on="ID", right_on="ID")
train_full.head()
test_full = test_variants_df.merge(test_text_df, how="inner", left_on="ID", right_on="ID")
test_full.head()
val_full = val_variants_df.merge(val_text_df, how="inner", left_on="ID", right_on="ID")
val_full.head()

train_full = train_full.drop("Text", axis=1)
test_full = test_full.drop("Text", axis=1)
val_full = val_full.drop("Text", axis=1)

print(sum(train_full["Text_count"]))
print(sum(test_full["Text_count"]))
print(sum(val_full["Text_count"]))


from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(train_text_df['Text_processed'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="black", max_words=5000, contour_width=5, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()

count_grp = train_full.groupby('Class')["Text_count"]
count_grp.describe()

plt.figure(figsize=(12,8))
gene_count_grp = train_full.groupby('Gene')["Text_count"].sum().reset_index()
sns.violinplot(x="Class", y="Text_count", data=train_full, inner=None,palette="Spectral")
sns.swarmplot(x="Class", y="Text_count", data=train_full, color="w", alpha=.5);
plt.ylabel('Text Count', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title("Text length distribution", fontsize=18)
plt.show()


fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20,16))

for i in range(3):
    for j in range(3):
        gene_count_grp = train_full[train_full["Class"]==((i*3+j)+1)].groupby('Gene')["Text_count"].mean().reset_index()
        sorted_gene_group = gene_count_grp.sort_values('Text_count', ascending=False)
        sorted_gene_group_top_7 = sorted_gene_group[:7]
        sns.barplot(x="Gene", y="Text_count", data=sorted_gene_group_top_7, ax=axs[i][j],palette="Spectral")


# # Vector Representation

train_full["Text_processed"]
test_full["Text_processed"]
val_full["Text_processed"]

# ## Word2Vec

# ### CBOW and Skip Gram


# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

Word2Vec_dim = 200
train = train_full["Text_processed"].to_string()
test = test_full["Text_processed"].to_string()
val = val_full["Text_processed"].to_string()

# Replaces escape character with space
f = train.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
    
    data.append(temp)

    
# Replaces escape character with space
f2 = test.replace("\n", " ")

data2 = []

# iterate through each sentence in the file
for i in sent_tokenize(f2):
    temp2 = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp2.append(j.lower())
    
    data2.append(temp2)

     
# Replaces escape character with space
f3 = val.replace("\n", " ")

data3 = []

# iterate through each sentence in the file
for i in sent_tokenize(f3):
    temp3 = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp3.append(j.lower())
    
    data3.append(temp3)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1,vector_size = Word2Vec_dim, window = 5)
model2 = gensim.models.Word2Vec(data2, min_count = 1,vector_size = Word2Vec_dim, window = 5)
model3 = gensim.models.Word2Vec(data3, min_count = 1,vector_size = Word2Vec_dim, window = 5)


#Create Skip Gram model
model4 = gensim.models.Word2Vec(data, min_count = 1, vector_size = Word2Vec_dim, window = 5, sg = 1)
model5 = gensim.models.Word2Vec(data2, min_count = 1, vector_size = Word2Vec_dim, window = 5, sg = 1)
# model6 = gensim.models.Word2Vec(data2, min_count = 1, vector_size = Word2Vec_dim, window = 5, sg = 1)

# model1.save("word2vec.model1_CBOW")
# model2.save("word2vec.model2_CBOW")
# model3.save("word2vec.model3_CBOW")
# model4.save("word2vec.model4_SkipGram")
# model5.save("word2vec.model5_SkipGram")
# model6.save("word2vec.model6_SkipGram")


# ### Average feature vector


def avg_feature_vector(sentence, model, num_features):
    words = sentence.replace('\n'," ").replace(',',' ').replace('.'," ").split()
    feature_vec = np.zeros((num_features,),dtype="float32")
    i=0
    for word in words:
        try:
            feature_vec = np.add(feature_vec, model[word])
        except KeyError as error:
            feature_vec 
            i = i + 1
    if len(words) > 0:
        feature_vec = np.divide(feature_vec, len(words)- i)
    return feature_vec


train_word2vec1 = np.zeros((len(train_full),Word2Vec_dim),dtype="float32")
test_word2vec1 = np.zeros((len(test_full),Word2Vec_dim),dtype="float32")
val_word2vec1 = np.zeros((len(val_full),Word2Vec_dim),dtype="float32")

for i in range(len(train_full)):
    train_word2vec1[i] = avg_feature_vector(train_full["Text_processed"][i],model1.wv, Word2Vec_dim)
    
for i in range(len(test_full)):
    test_word2vec1[i] = avg_feature_vector(test_full["Text_processed"][i],model1.wv, Word2Vec_dim) 
    
for i in range(len(val_full)):
    val_word2vec1[i] = avg_feature_vector(val_full["Text_processed"][i],model1.wv, Word2Vec_dim) 


train_word2vec2 = np.zeros((len(train_full),Word2Vec_dim),dtype="float32")
test_word2vec2 = np.zeros((len(test_full),Word2Vec_dim),dtype="float32")
val_word2vec2 = np.zeros((len(val_full),Word2Vec_dim),dtype="float32")

for i in range(len(train_full)):
    train_word2vec2[i] = avg_feature_vector(train_full["Text_processed"][i],model4.wv, Word2Vec_dim)
    
for i in range(len(test_full)):
    test_word2vec2[i] = avg_feature_vector(test_full["Text_processed"][i],model4.wv, Word2Vec_dim)
    
for i in range(len(val_full)):
    val_word2vec2[i] = avg_feature_vector(val_full["Text_processed"][i],model4.wv, Word2Vec_dim) 


train_word2vec1 = pd.DataFrame(train_word2vec1)
train_word2vec1
test_word2vec1 = pd.DataFrame(test_word2vec1)
test_word2vec1
val_word2vec1 = pd.DataFrame(val_word2vec1)
val_word2vec1
train_word2vec2 = pd.DataFrame(train_word2vec2)
train_word2vec2
test_word2vec2 = pd.DataFrame(test_word2vec2)
test_word2vec2
val_word2vec2 = pd.DataFrame(val_word2vec2)
val_word2vec2


# # Modeling

# ## LightGBM

lbl = preprocessing.LabelEncoder()
data1 = train_word2vec1
target = lbl.fit_transform(train_full["Class"].astype(str))#将提示的包含错误数据类型这一列进行转换
X_train1, X_test1, y_train1, y_test1 = train_test_split(data1,target,test_size=0.3, random_state = 42)
print("Model1: Train data length:", len(X_train1))
print("Model1: Test data length:", len(X_test1))
btime = datetime.now() 
lgb_train1 = lgb.Dataset(X_train1, y_train1)
lgb_eval1 = lgb.Dataset(X_test1, y_test1, reference=lgb_train1)

params = {
    'task':'train',
    'boosting_type':'gbdt',  
    'objective': 'multiclass',  
    'num_class': 9,  
    'metric': 'multi_error',  
    'num_leaves': 500,  
    'min_data_in_leaf': 100,  
    'learning_rate': 0.1,  
    'feature_fraction': 0.9,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 5,  
    'lambda_l1': 0.4,  
    'lambda_l2': 0.5,  
    'min_gain_to_split': 0.2,  
    'verbose': -1, 
}

gbm1 = lgb.train(params, lgb_train1, num_boost_round=1000, valid_sets=lgb_eval1,
    verbose_eval = 50, early_stopping_rounds=300)
print('all tasks done. total time used:%s s.\n\n'%((datetime.now() - btime).total_seconds()))
gbm1.save_model('model1_CBOW.txt')
gbm1 = lgb.Booster(model_file='model1_CBOW.txt')
y_pred_pa1 = gbm1.predict(X_test1)
y_test_oh1 = label_binarize(y_test1, classes= [1,2,3,4,5,6,7,8,9])
#y_pred_lightGBM1 = [list(x).index(max(x)) for x in y_prob1]
print('auc：', roc_auc_score(y_test_oh1, y_pred_pa1, average='micro'))

y_pred1 = y_pred_pa1.argmax(axis=1)
confusion_matrix(y_test1, y_pred1)

precision_score(y_test1, y_pred1,average='micro')
recall_score(y_test1, y_pred1,average='micro')
f1_score(y_test1, y_pred1,average='micro')

print(classification_report(y_test1, y_pred1))

pred_lightgbm_cbow = gbm1.predict(test_word2vec1)
#pred_lightgbm_cbow = [list(x).index(max(x)) for x in pred_lightgbm_cbow]
#pred_lightgbm_cbow = pd.get_dummies(np.array(pred_lightgbm_cbow) + 1)
pred_lightgbm_cbow

# pred_lightgbm_skg = gbm2.predict(test_word2vec2)
# # pred_lightgbm_skg = [list(x).index(max(x)) for x in pred_lightgbm_skg]
# # pred_lightgbm_skg = pd.get_dummies(np.array(pred_lightgbm_skg) + 1)
# pred_lightgbm_skg


# ## RNN



NUM_CLASS=9
VOCABULARY_SIZE = 10000
SEQUENCE_LENGTH= 2000
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
tokenizer.fit_on_texts(train_full["Text_processed"])
vocab = tokenizer.word_index

# Training
training = train_full.sample(frac=1) # shuffle data first
training_input = tokenizer.texts_to_sequences(training['Text_processed'].astype(str))
training_input_r = [list(reversed(x)) for x in training_input]
training_input_begin = pad_sequences(training_input, maxlen=SEQUENCE_LENGTH)
training_input_end = pad_sequences(training_input_r, maxlen=SEQUENCE_LENGTH)
training_output = pd.get_dummies(training['Class']).values

# Testing
testing_input = tokenizer.texts_to_sequences(test_full['Text_processed'].astype(str))
testing_input_r = [list(reversed(x)) for x in testing_input]
testing_input_begin = pad_sequences(testing_input, maxlen=SEQUENCE_LENGTH)
testing_input_end = pad_sequences(testing_input_r, maxlen=SEQUENCE_LENGTH)

# Validation
val_input = tokenizer.texts_to_sequences(val_full['Text_processed'].astype(str))
val_input_r = [list(reversed(x)) for x in val_input]
val_input_begin = pad_sequences(val_input, maxlen=SEQUENCE_LENGTH)
val_input_end = pad_sequences(val_input_r, maxlen=SEQUENCE_LENGTH)
val_output = pd.get_dummies(val_full['Class']).values

print("Training set shape:",training_input_begin.shape, training_input_end.shape, training_output.shape)
print("Testing set shape:",testing_input_begin.shape, testing_input_end.shape)
print("Validation set shape:",val_input_begin.shape, val_input_end.shape, val_output.shape)

# Add gene and variation to predictor
gene_label = LabelEncoder()
ALL_Genes = np.concatenate([train_full['Gene'], val_full['Gene'], test_full['Gene']])
ALL_Variations = np.concatenate([train_full['Variation'], val_full['Variation'], test_full['Variation']])
ALL_Variations = np.asarray([v[0]+v[-1] for v in ALL_Variations])
print ("The number of unique genes: ", len(np.unique(ALL_Genes)))
print ("The number of unique variations:", len(np.unique(ALL_Variations)))


len_train = len(training_input)
len_validation = len(val_input)
len_test = len(testing_input)
print("The length of training input:", len_train)
print("The length of testing input:", len_test)
print("The length of validation input:", len_validation)


gene_encoded = pd.get_dummies(ALL_Genes).values
variation_encoded = pd.get_dummies(ALL_Variations).values
training_input_gene = gene_encoded[:len_train]
training_input_variation = variation_encoded[:len_train]
testing_input_gene = gene_encoded[-len_test:]
testing_input_variation = variation_encoded[-len_test:]
val_input_gene = gene_encoded[len_train:-len_test]
val_input_variation = variation_encoded[len_train:-len_test]
print (len(training_input_gene))
print (len(testing_input_gene))
print (len(val_input_gene))


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


# ### Attention with context

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape = (input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape = (input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

#The number of unique genes:  1522
#The number of unique variations: 347


# ### GRU + Bidirectional

Embedding_dim = 200
lstm_out = 64

# Model saving callback
ckpt_callback = ModelCheckpoint('keras_model', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

input_sequence_begin = Input(shape=(training_input_begin.shape[1],))
input_sequence_end = Input(shape=(training_input_end.shape[1],))
input_gene = Input(shape=(training_input_gene.shape[1],))
input_variant = Input(shape=(training_input_variation.shape[1],))

merged = concatenate([input_gene, input_variant])
dense = Dense(32, activation='sigmoid')(merged)

embeds_begin = Embedding(VOCABULARY_SIZE, Embedding_dim, input_length = SEQUENCE_LENGTH)(input_sequence_begin)
embeds_out_begin = Bidirectional(GRU(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))(embeds_begin)
attention_begin = AttentionWithContext()(embeds_out_begin)

embeds_end = Embedding(VOCABULARY_SIZE, Embedding_dim, input_length = SEQUENCE_LENGTH)(input_sequence_end)
embeds_out_end = Bidirectional(GRU(lstm_out, recurrent_dropout=0.2, dropout=0.2, return_sequences=True))(embeds_end)
attention_end = AttentionWithContext()(embeds_out_end)

merged2 = concatenate([attention_begin, attention_end, dense])
dense2 = Dense(9,activation='softmax')(merged2)

model_RNN = Model(inputs=[input_sequence_begin, input_sequence_end, input_gene, input_variant], outputs=dense2)
model_RNN.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_RNN.summary())


model_RNN.fit([training_input_begin, training_input_end, training_input_gene, training_input_variation], 
              training_output, epochs=10, batch_size=32,
              validation_data=([val_input_begin,val_input_end,val_input_gene,val_input_variation], val_output),
               callbacks=[ckpt_callback])


probas = model_RNN.predict([val_input_begin, val_input_end, val_input_gene, val_input_variation])
pred_indices = np.argmax(probas, axis=1)
classes = np.array(range(1, 10))
preds = classes[pred_indices]
print('Log loss: {}'.format(log_loss(classes[np.argmax(val_output, axis=1)], probas)))
print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(val_output, axis=1)], preds)))
skplt.plot_confusion_matrix(classes[np.argmax(val_output, axis=1)], preds)

model_RNN.fit([
    np.concatenate([training_input_begin, val_input_begin]), 
    np.concatenate([training_input_end,val_input_end]), 
    np.concatenate([training_input_gene, val_input_gene]), 
    np.concatenate([training_input_variation, val_input_variation])],
    np.concatenate([training_output,val_output]), 
    epochs=3, batch_size=32, callbacks=[ckpt_callback])


probas = model_RNN.predict([testing_input_begin, testing_input_end, testing_input_gene, testing_input_variation])

# Submission
pred_lightgbm_cbow = pd.DataFrame(pred_lightgbm_cbow)
pred_lightgbm_cbow
NEWsubmission_df1 = pred_lightgbm_cbow
NEWsubmission_df1.to_csv("NEWsubmission_lightGBM.csv",index=False)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = test_full['ID']
submission_df
submission_df.to_csv("NEWsubmission_rnn2000.csv",index=False)