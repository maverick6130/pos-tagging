from model import Model
from utils import clean_corpus, save_classification_results

class POSTagger_Neural (Model):
    
    def __init__(self) -> None:
        super().__init__("neural")
        self.init = False
    
    
    def train(self, corpus) -> None:
        pass
                
    
    def test(self, corpus, result_id) -> None:
        pass

    
    def save(self, model_id) -> None:
        pass
        
        
    def load(self, model_id) -> None:
        pass
            
    
    def predict_tags(self, sent):
        pass