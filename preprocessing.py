import pandas as pd
import re
import csv


def load_data(path):
    preprocessed_data = []
    df = pd.read_csv(path, encoding='latin1')
    for index, row in df.iterrows():
        post = row['Comment']

        post = post.lower()
        post = re.sub(r'ies( |$)', 'y ', post, flags=re.MULTILINE)
        post = re.sub(r's( |$)', ' ', post, flags=re.MULTILINE)
        post = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', 'URL', post, flags=re.MULTILINE)
        post = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', post, flags=re.MULTILINE)
        post = re.sub(r'[\w\.-]+@[\w\.-]+', 'EMAIL', post, flags=re.MULTILINE)

        post = re.sub(r' [0-9]+ ', ' DD ', post, flags=re.MULTILINE)
        post = re.sub(r'<\S*>', '', post, flags=re.MULTILINE)
        post = post.replace(',', ' ')
        post = post.replace('\r', '').replace('\n', '')
        post = post.replace("I've", "I have")
        post = post.replace("I'm", "I am")
        post = post.replace("i've", "I have")
        post = post.replace("it's", "it is")
        post = post.replace("It's", "It is")
        post = post.replace("wasn't", "was not")
        post = post.replace("aren't", "are not")
        post = post.replace("isn't", "is not")
        post = post.replace("weren't", "were not")
        post = post.replace("don't", "do not")
        post = post.replace("didn't", "did not")
        post = post.replace("I'll", "I will")
        post = post.replace("won't", "will not")
        post = post.replace("I'd", "I would")
        post = post.replace("you're", "you are")
        post = post.replace("you'll", "you will")
        post = post.replace("they'll", "they will")
        post = post.replace("cant", "cannot")
        post = post.replace("dont", "do not")
        post = post.replace("didnt", "did not")
        post = post.replace("couldn't", "could not")
        post = post.replace("cann't", "can not")
        post = post.replace("/r/", "")
        post = post.replace("\\xa0", " ")
        post = post.replace("\\xc2", " ")
        post = post.replace(" u ", " you ")
        post = post.replace(" em ", " them ")
        post = post.replace(" da ", " the ")
        post = post.replace(" yo ", " you ")
        post = post.replace(" ur ", " you ")
        post = post.replace(" y ", " why ")
        post = post.replace(" im ", " i am ")
        post = post.replace("ain't", "is not")
        post = post.replace("'ll", " will")
        post = post.replace("'t", " not")
        post = post.replace("'ve", " have")
        post = post.replace("'s", " is")
        post = post.replace("'re", " are")
        post = post.replace("'d", " would")

        # post = post.replace("tard ", " ")
        # post = re.sub(r'ing( |$)', ' ', post, flags=re.MULTILINE)

        new_row = (row['ID'], post, row['Label'])
        preprocessed_data.append(new_row)

    _path = path.split('.')
    new_path = _path[0] + '_preprocessed.' + _path[1]
    labels = ['ID', 'Comment', 'Label']
    df = pd.DataFrame.from_records(preprocessed_data, columns=labels)
    df.to_csv(new_path, encoding='latin1')



load_data('resources/english/agr_en_sm_test.csv')