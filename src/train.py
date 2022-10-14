#Setup model selection
from hmm_tagger import POSTagger_HMM
from hmm_wv_tagger import POSTagger_HMM_WV
from neural_tagger import POSTagger_Neural
import argparse
parser = argparse.ArgumentParser('Train a POS tagging model with 5-fold validation on the brown corpus. Save a model trained on the complete corpus.')
parser.add_argument('--model', required=True, type=str, choices={'hmm', 'hmm_w2v', 'neural'}, help='Type of POS tagging model')
args = parser.parse_args()
if args.model == 'hmm':
    get_model = lambda : POSTagger_HMM()
if args.model == 'hmm_w2v':
    get_model = lambda : POSTagger_HMM_WV()
if args.model == 'neural':
    get_model = lambda : POSTagger_Neural()
    
#Setup corpus
import nltk
nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown
corpus = brown.tagged_sents(tagset='universal')

#Training and cross validation
k = 1
from sklearn.model_selection import KFold
for train_idx, test_idx in KFold(5, shuffle=True, random_state=335).split(corpus):
    print(f'Performing 5-fold cross validation keeping subset {k} away')
    train_set = [ corpus[i] for i in train_idx ]
    test_set = [ corpus[i] for i in test_idx ]
    tagger = get_model()
    tagger.train(train_set)
    tagger.test(test_set, k)
    k += 1
from utils import kfold_validation_results
kfold_validation_results(get_model().results_dir)

#Final model
print('Training and saving the final prediction model')
tagger = get_model()
tagger.train(corpus)
tagger.save(0)
tagger = get_model()
tagger.load(0)
sent = ['i', 'bank', 'on', 'you', 'to', 'deposit', 'the', 'money', 'in', 'the', 'bank', '.']
print(' '.join([ f'{w}:{t}' for w,t in zip(sent, tagger.predict_tags(sent)) ]))