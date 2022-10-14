from model import Model
from utils import clean_corpus, save_classification_results

from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os
import json

class POSTagger_HMM (Model):
    
    def __init__(self) -> None:
        super().__init__("hmm")
        self.init = False
    
    
    def train(self, corpus) -> None:
        corpus = clean_corpus(corpus)
        self.tags = sorted(set([ t for sent in corpus for _,t in sent ]))
        self.epsilon = 1/np.sum([ len(sent) for sent in corpus ])
        n = len(self.tags)
        self.tag_idx = { self.tags[i] : i for i in range(n) }

        self.transition_prob = np.zeros((n+1,n+1))
        self.occur_prob = [defaultdict(int) for _ in range(n)]
        total_occur = [0 for _ in range(n)] + [len(corpus)]
        for sent in tqdm(corpus, desc="Training HMM Model"):
            prev = n
            for w,t in sent:
                tag = self.tag_idx[t]
                self.occur_prob[tag][w] += 1
                self.transition_prob[prev][tag] += 1
                total_occur[tag] += 1
                prev = tag
        
        with np.errstate(divide='ignore'):
            self.transition_prob = np.log(self.transition_prob.T/np.sum(self.transition_prob, axis=1)).T
        self.occur_prob = [ { w : np.log(count/total_occur[i]) for w, count in self.occur_prob[i].items() } for i in range(n) ]
        self.init = True

    
    def save(self, model_id) -> None:
        assert(self.init)
        with open(os.path.join(self.model_dir, f'model_{model_id}.json'), 'w') as file:
            model = {}
            model['tags'] = self.tags
            model['tag_idx'] = self.tag_idx
            model['transition_prob'] = self.transition_prob.tolist()
            model['occur_prob'] = self.occur_prob
            model['epsilon'] = self.epsilon
            json.dump(model, file, indent=2)
        
        
    def load(self, model_id) -> None:
        with open(os.path.join(self.model_dir, f'model_{model_id}.json'), 'r') as file:
            model = json.load(file)
            self.tags = model['tags']
            self.tag_idx = model['tag_idx']
            self.transition_prob = np.array(model['transition_prob'])
            self.occur_prob = model['occur_prob']
            self.epsilon = model['epsilon']
        self.init = True
            
    
    def predict_tags(self, sent):
        assert(self.init)
        n = len(self.tags)
        best_seq = [ [i] for i in range(n+1) ]
        log_best_prob = np.array([-np.inf for _ in range(n)] + [0.0])
        for word in sent:
            log_emit = np.array([ self.occur_prob[i][word] if word in self.occur_prob[i] else np.log(self.epsilon) for i in range(n) ] + [-np.inf])
            log_end_prob = ((self.transition_prob.T + log_best_prob).T + log_emit).T
            log_best_prob = np.max(log_end_prob, axis=1)
            seq_picked = np.argmax(log_end_prob, axis=1)
            best_seq = [ best_seq[seq_picked[i]] + [i] for i in range(n+1) ]
        best_of_best_idx = np.argmax(log_best_prob)
        return [ self.tags[tag] for tag in best_seq[best_of_best_idx][1:] ]