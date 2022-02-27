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
 
if not os.path.exists(f"preprocessed_{train_data_path.split('/')[-1].split('.')[0]}.csv"):
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

            temp = pd.DataFrame({"id":[text_id], "text":[text], "category":[category]})
            train_data = pd.concat([train_data, temp], ignore_index=True)
    print("len of df: ", len(train_data))
    train_data.to_csv(f"preprocessed_{train_data_path.split('/')[-1].split('.')[0]}.csv")
else:
    train_data = pd.read_csv(f"preprocessed_{train_data_path.split('/')[-1].split('.')[0]}.csv")
    print("saved len of train data: ", len(train_data))
    # print(train_data.head())

if not os.path.exists(f"preprocessed_{test_data_path.split('/')[-1].split('.')[0]}.csv"):
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
    test_data.to_csv(f"preprocessed_{test_data_path.split('/')[-1].split('.')[0]}.csv")
else:
    test_data = pd.read_csv(f"preprocessed_{test_data_path.split('/')[-1].split('.')[0]}.csv")
    print("saved len of train data: ", len(test_data))
    # print(train_data.head())

# exit(1)
# # if not os.path.exists(f"preprocessed_training_{train_data_path.split('/')[-2]}.csv"):
# #     # loops through files in the directory
# #     for filename in os.listdir(train_data_path):
# #         text = ""
# #         # loops through lines of the article
# #         with open(os.path.join(train_data_path, filename), 'r') as f:
# #             for line in f:
# #                 line = f.readline()
# #                 line = text_preprocess(line)
# #                 text = text + " " + line

# #             temp = pd.DataFrame({"id":[filename.split('.')[0]], "text":[text]})
# #             train_data = pd.concat([train_data, temp], ignore_index=True)
# #     # print(train_data.head())
# #     print("len of df: ", len(train_data))
# #     train_data.to_csv(f"preprocessed_training_{train_data_path.split('/')[-2]}.csv")
# # else:
# #     train_data = pd.read_csv(f"preprocessed_training_{train_data_path.split('/')[-2]}.csv")
# #     print("saved len of train data: ", len(train_data))


# if not os.path.exists(f"preprocessed_testing_{test_data_path.split('/')[-2]}.csv"):
#     # loops through files in the directory
#     for filename in os.listdir(test_data_path):
#         text = ""
#         # loops through lines of the article
#         with open(os.path.join(test_data_path, filename), 'r') as f:
#             for line in f:
#                 line = f.readline()
#                 line = text_preprocess(line)
#                 text = text + " " + line

#             temp = pd.DataFrame({"id":[filename.split('.')[0]], "text":[text]})
#             test_data = pd.concat([test_data, temp])
#     print("len of test df: ", len(test_data))
#     train_data.to_csv(f"preprocessed_testing_{test_data_path.split('/')[-2]}.csv")
# else:
#     test_data = pd.read_csv(f"preprocessed_testing_{test_data_path.split('/')[-2]}.csv")
#     print("saved len of test data: ", len(test_data))


## Calculate TF-IDF weights
# tfidf_vectorizer = TfidfVectorizer(use_idf=True)
# train_tfidf = tfidf_vectorizer.fit_transform(train_data["text"])
# print("size: ", train_tfidf.shape)
# print("feature size :", len(tfidf_vectorizer.get_feature_names_out()))
# print(tfidf_vectorizer.get_feature_names_out())
# test_tfidf = tfidf_vectorizer.fit_transform(test_data)

# Baysian
# 
