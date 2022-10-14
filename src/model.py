class Model:
    def __init__(self, name) -> None:
        self.name = name
        self.results_dir = f'results/{name}/'
        self.model_dir = f'model/{name}/'