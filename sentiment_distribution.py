import os
import json
import numpy as np

def extract_sentiment_distribution_feature(mode='train'):
    json_path = 'stanford_sentiment_analysis_vectors/{}/'.format(mode)

    files = os.listdir(json_path)
    sent_dist_dict = {}

    for file in files:
        try:
            if 'facebook_corpus_msr_1493901' in file:
                print "hereee"
                print file[:-5]
            very_neg = []
            neg = []
            neut = []
            pos = []
            very_pos = []

            data = json.load( open(json_path + file) )
            sentences = data['sentences']
            # if len(sentences) == 1:
            #     continue

            for sentence in sentences:
                sent_dist = sentence['sentimentDistribution']
                very_neg.append(sent_dist[0])
                neg.append(sent_dist[1])
                neut.append(sent_dist[2])
                pos.append(sent_dist[3])
                very_pos.append(sent_dist[4])

            means = []
            stds = []

            means.append(np.mean(very_neg))
            means.append(np.mean(neg))
            means.append(np.mean(neut))
            means.append(np.mean(pos))
            means.append(np.mean(very_pos))

            stds.append(np.std(very_neg))
            stds.append(np.std(neg))
            stds.append(np.std(neut))
            stds.append(np.std(pos))
            stds.append(np.std(very_pos))


            sent_dist_dict[file[:-5]] = {'means': means, 'stds': stds}
            if 'facebook_corpus_msr_1493901' in file:
                print file[:-5]

        except:
            print(file)

    json.dump(sent_dist_dict, open('stanford_sentiment_analysis/{}_dict.json'.format(mode), 'w'))

extract_sentiment_distribution_feature('dev')