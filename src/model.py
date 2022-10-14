from tqdm import tqdm
from utils import save_classification_results, clean_corpus
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self, name) -> None:
        self.name = name
        self.results_dir = f'results/{name}/'
        self.model_dir = f'model/{name}/'


    def test(self, corpus, result_id) -> None:
        assert(self.init)
        corpus = clean_corpus(corpus)
        true_label = []
        pred_label = []
        accuracy = 0
        total = 0
        for sent in (pbar := tqdm(corpus, desc=f'Testing {self.name} model')):
            y_true = [ t for _,t in sent ]
            y_pred = self.predict_tags([w for w,_ in sent])
            assert(len(y_true) == len(y_pred))
            accuracy *= total
            accuracy += accuracy_score(y_true, y_pred)*len(y_true)
            total += len(y_true)
            accuracy /= total
            true_label += y_true
            pred_label += y_pred
            pbar.set_postfix(accuracy = accuracy)
        save_classification_results(true_label, pred_label, self.results_dir, result_id)