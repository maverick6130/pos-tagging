from hmm_tagger import POSTagger_HMM
from hmm_wv_tagger import POSTagger_HMM_VW
from neural_tagger import POSTagger_Neural

import argparse
import string

parser = argparse.ArgumentParser('Train a POS tagging model with 5-fold validation on the brown corpus. Save a model trained on the complete corpus.')
parser.add_argument('--model', required=True, type=str, choices={'hmm', 'hmm_w2v', 'neural'}, help='Type of POS tagging model')
args = parser.parse_args()
if args.model == 'hmm':
    get_model = lambda : POSTagger_HMM()
if args.model == 'hmm_w2v':
    get_model = lambda : POSTagger_HMM_VW()
if args.model == 'neural':
    get_model = lambda : POSTagger_Neural()
    
tagger = get_model() 
tagger.load(0)
print(f'Tagger {args.model} loaded. You can interactively tag sentences now.')
while(True):
    consent = input('Continue?')
    if consent in ('no', 'No', 'NO', 'n', 'N'):
        break
    if consent in ('yes', 'Yes', 'YES', 'y', 'Y'):
        words = input().split()
        sent = []
        for w in words:
            clean_w = w.lower().translate(str.maketrans('','',string.punctuation))
            if clean_w != '':
                sent += [clean_w]
            else:
                sent += [w]
        if sent[-1] != '.':
            sent += ['.']
        tags = tagger.predict_tags(sent)
        print(" ".join(tags))