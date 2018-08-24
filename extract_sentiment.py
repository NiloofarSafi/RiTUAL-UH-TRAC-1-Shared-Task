from pycorenlp import StanfordCoreNLP
import csv
import pandas as pd
import json
import re


with open('missed_train.txt') as f:
    content1 = f.readlines()

missed_train = set([x.split('.')[0].strip() for x in content1])



with open('missed_dev.txt') as f:
    content = f.readlines()

missed_dev = set([x.split('.')[0].strip() for x in content])



nlp = StanfordCoreNLP('http://localhost:9000')


def load_data(path):
    doc_id = []
    X = []
    y = []
    df = pd.read_csv(path, encoding='latin1')
    for index, row in df.iterrows():
        doc_id.append(row['ID'])
        st = row['Comment'].encode('ascii', 'ignore').decode('ascii')
        X.append(re.sub(r"((www\.[^\s]+)|(https?:\/\/[^\s]+))", '', st, flags=re.MULTILINE))
        if row['Label'] == "NAG":
            y.append(1)
        else:
            y.append(0)
    return doc_id, X, y


data_path = "resources/english/agr_en_train.csv"
doc_id, X, y = load_data(data_path)

print len(doc_id)
print doc_id



flag = True
for i in range(len(doc_id)):
    # if doc_id[i] == 'facebook_corpus_msr_325604' or doc_id[i] == 'facebook_corpus_msr_374350' or doc_id[i] == 'facebook_corpus_msr_464484':
    if doc_id[i] == 'facebook_corpus_msr_1493901':

        json_content = nlp.annotate(str(X[i]),
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 15000000,
                   })

        print doc_id[i]
        print json_content

        path = 'stanford_sentiment_analysis_vectors/train/' + doc_id[i] + '.json'
        with open(path, 'w') as outfile:
            json.dump(json_content, outfile)


