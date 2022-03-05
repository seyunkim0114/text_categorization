import os
import os.path
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from control.config import args

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# nltk.download('punkt')

train_data_path = args.train_data_path
test_data_path = args.test_data_path
train_data = pd.DataFrame()
test_data = pd.DataFrame()

lemmatizer = WordNetLemmatizer()

def text_preprocess(line):
    line = line.lower()
    line = [i for i in line.split() if i not in set(stopwords.words('english'))]
    line = ' '.join(line)
    line = word_tokenize(line)
    line = nltk.pos_tag(line)
    line = [lemmatizer.lemmatize(tag[0]) for idx, tag in enumerate(line)]
    line = ' '.join(line)

    return line
 
if not os.path.exists("preprocessed"):
    os.mkdir("preprocessed")
else:
    pass

if not os.path.exists(f"preprocessed/preprocessed_{train_data_path.split('/')[-1].split('.')[0]}.csv"):
    with open(train_data_path, 'r') as f:
        for path in f:
            text = ""
            # path = f.readline()
            path_to_text = path.split(" ")[0]
            text_id = path_to_text.split(' ')[0].split('/')[-1].split('.')[0]
            category = path.split(" ")[-1]
            # print(path_to_text[2:])
            # print(os.path.join(os.path.dirname(train_data_path), path_to_text[2:]))
            with open(os.path.join(os.path.dirname(train_data_path), path_to_text[2:]), 'r') as file:
                for line in file:
                    # line = file.readline()
                    line = text_preprocess(line)
                    text = text + " " + line
            tokenized = nltk.tokenize.word_tokenize(text)
            fdist = nltk.FreqDist(tokenized).keys()
            temp = pd.DataFrame({"id":[text_id], "text":[text], "fdist":[fdist], "category":[category]})
            train_data = pd.concat([train_data, temp], ignore_index=True)
    print("len of df: ", len(train_data))
    train_data.to_csv(f"preprocessed/preprocessed_{train_data_path.split('/')[-1].split('.')[0]}.csv")
else:
    train_data = pd.read_csv(f"preprocessed/preprocessed_{train_data_path.split('/')[-1].split('.')[0]}.csv")
    print("saved len of train data: ", len(train_data))
    # print(train_data.head())

if not os.path.exists(f"preprocessed/preprocessed_{test_data_path.split('/')[-1].split('.')[0]}.csv"):
    with open(test_data_path, 'r') as f:
        for path in f:
            text = ""
            # path = f.readline()
            path_to_text = path.split(" ")[0]
            text_id = path_to_text.split(' ')[0].split('/')[-1].split('.')[0]
            category = path.split(" ")[-1]
            # print(path_to_text[2:])
            # print(os.path.join(os.path.dirname(test_data_path), path_to_text[2:]))
            with open(os.path.join(os.path.dirname(test_data_path), path_to_text[2:]), 'r') as file:
                for line in file:
                    # line = file.readline()
                    line = text_preprocess(line)
                    text = text + " " + line

            temp = pd.DataFrame({"id":[text_id], "text":[text], "category":[category]})
            test_data = pd.concat([test_data, temp], ignore_index=True)
    print("len of df: ", len(test_data))
    test_data.to_csv(f"preprocessed/preprocessed_{test_data_path.split('/')[-1].split('.')[0]}.csv")
else:
    test_data = pd.read_csv(f"preprocessed/preprocessed_{test_data_path.split('/')[-1].split('.')[0]}.csv")
    print("saved len of train data: ", len(test_data))
    # print(train_data.head())


# Label encode train documents and get doc statistics
total_train_documents = len(train_data)
train_doc_category = train_data["category"].unique()
le = LabelEncoder()
labels = le.fit_transform(train_data["category"])
# train_data["labels"] = labels

for i in range(max(labels)):
    _, category_freq_train = np.unique(labels, return_counts=True)
train_classes = {v:k for v,k in enumerate(le.classes_)}


def get_maximum_likelihood_estimate(t, c, data):
    '''
    Input : t : a word in a document (string)
            c : category (int or string)
            data : train or test data (dataframe)
    Output : logP(t|c) 

    Returns maximum likelihood estimate (probability of seeing t in category c)
    '''
    category_c = data.loc[data["labels"] == c]
    print(category_c)
    count_docs_contain_t = 0

    for doc in category_c:
        if t in doc["text"]:
            count_docs_contain_t += 1

    return count_docs_contain_t / len(category_c)

prob = get_maximum_likelihood_estimate("activity", 0, train_data)

    




# Bayesian Text Categorization

category_prob_train = category_freq_train / total_train_documents # P(c)
all_words = set(train_data["fdist"])
# print(len(all_words))

# for i in range(max(labels)):
#     docs_in_i = train_data.index[train_data["labels"] == i].tolist()
#     for j in docs_in_i:
#         if 



# print(labels)
# train_label = ohe.fit_transform(labels).toarray()
# print("0: ", len(train_label))
# print("1: ", train_label.shape)
# print("2: ", train_label[0])