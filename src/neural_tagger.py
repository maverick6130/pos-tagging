from model import Model
from utils import clean_corpus, save_classification_results

import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score

import time
import fasttext.util
print("Loading Word2Vec model :", end=' ')
t0 = time.time()
fasttext.util.download_model('en', if_exists='ignore')  # English
w2vmodel = fasttext.load_model('cc.en.300.bin')
t1 = time.time()
print(f'Done. That took {t1-t0}s')

import torch
from torch import nn
class NeuralNet(nn.Module):
    def __init__(self, in_sz, hidden_sz, out_sz) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_sz, hidden_sz)
        self.fc2 = nn.Linear(hidden_sz, out_sz)

    def forward(self, x):
        x = self.fc1(x)
        x = x.tanh()
        x = self.fc2(x)
        return x

def get_nn_input_single(sent):
    nn_input = []
    prev = w2vmodel['^']
    for w in sent:
        embed = w2vmodel[w]
        nn_input += [np.hstack((prev, embed))]
        prev = embed
    return torch.tensor(np.array(nn_input))
        

def get_nn_input(corpus):
    nn_input = []
    for sent in tqdm(corpus, desc="Building inputs"):
        prev = w2vmodel['^']
        for w,_ in sent:
            embed = w2vmodel[w]
            nn_input += [np.hstack((prev, embed))]
            prev = embed
    return torch.tensor(np.array(nn_input))

class POSTagger_Neural (Model):
    
    def __init__(self) -> None:
        super().__init__("neural")
        self.init = False

    
    def train(self, corpus, max_epoch = 100) -> None:
        corpus = clean_corpus(corpus)
        self.tags = sorted(set([ t for sent in corpus for _,t in sent ]))
        n = len(self.tags)
        self.tag_idx = { self.tags[i] : i for i in range(n) }

        self.model = NeuralNet(600, 200, n)
        y = torch.tensor([ self.tag_idx[t] for sent in corpus for _,t in sent ])
        X = get_nn_input(corpus)

        optim = torch.optim.SGD(
            self.model.parameters(),
            lr = 2.0,
            momentum= 0.4
        )
        loss_fn = nn.CrossEntropyLoss()

        for epoch in (pbar := tqdm(range(1, max_epoch+1), desc="Training network")):
            Z = self.model(X)
            loss = loss_fn(Z,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())

        self.init = True

    def test(self, corpus, result_id) -> None:
        assert(self.init)
        corpus = clean_corpus(corpus)
        true_label = [ t for sent in corpus for _,t in sent ] 
        X = get_nn_input(corpus)
        Z = self.model(X).detach().numpy()
        pred_label = [ self.tags[x] for x in np.argmax(Z, axis=1) ]
        print(f'Testing accuracy : {accuracy_score(pred_label, true_label)}')
        save_classification_results(true_label, pred_label, self.results_dir, result_id)

    def save(self, model_id) -> None:
        assert(self.init)
        pkl_file = os.path.join(self.model_dir, f'nn_{model_id}.pkl')
        torch.save(self.model, pkl_file)
        with open(os.path.join(self.model_dir, f'model_{model_id}.json'), 'w') as file:
            model = {}
            model['tags'] = self.tags
            model['tag_idx'] = self.tag_idx
            model['nn_file'] = pkl_file
            json.dump(model, file, indent=2)
        
        
    def load(self, model_id) -> None:
        with open(os.path.join(self.model_dir, f'model_{model_id}.json'), 'r') as file:
            model = json.load(file)
            self.tags = model['tags']
            self.tag_idx = model['tag_idx']
            pkl_file = model['nn_file']
            self.model = torch.load(pkl_file)
        self.init = True
            
    
    def predict_tags(self, sent):
        assert(self.init)
        n = len(self.tags)
        X = get_nn_input_single(sent)
        Z = self.model(X).detach().numpy()
        return [ self.tags[i] for i in np.argmax(Z, axis=1) ]
