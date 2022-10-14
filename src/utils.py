from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import os
import json
import string
from tqdm import tqdm
from matplotlib import pyplot as plt
import warnings
from collections import defaultdict
import numpy as np

def clean_word(word):
    word = word.strip()
    clean_word = word.lower().translate(str.maketrans('','',string.punctuation))
    if clean_word:
        return clean_word
    return word


def clean_corpus(corpus):
    cleaned_corpus = [ [ ( clean_word(w), t ) for w,t in sent ] for sent in tqdm(corpus, desc="Cleaning Corpus") ]
    return cleaned_corpus


def save_classification_results(true_label, pred_label, dir, result_id) -> None:
    labels = sorted(set(true_label + pred_label))
    precision, recall, f1score, _ = precision_recall_fscore_support(
        true_label, pred_label, labels=labels,average='weighted', zero_division=0)
    _, _, f2score, _ = precision_recall_fscore_support(
        true_label, pred_label, labels=labels,average='weighted', beta=2, zero_division=0)
    _, _, f05score, _ = precision_recall_fscore_support(
        true_label, pred_label, labels=labels, average='weighted', beta=0.5, zero_division=0)
    per_class_precision, per_class_recall, per_class_fscore, _ = \
        precision_recall_fscore_support(true_label, pred_label, labels=labels, zero_division=0)
    confusion_mtx = confusion_matrix(true_label, pred_label, labels=labels, normalize='true')
    per_class_accuracy = confusion_mtx.diagonal()

    print(f'accuracy : {accuracy_score(true_label, pred_label)}')
    
    with open(os.path.join(dir, f'result_{result_id}.json'), 'w') as f:
        data = {}
        data['precision'] = precision
        data['recall'] = recall
        data['f1score'] = f1score
        data['f2score'] = f2score
        data['f0.5score'] = f05score
        data['classes'] = labels
        data['confusion_matrix'] = confusion_mtx.tolist()
        data['per_pos_accuracy'] = per_class_accuracy.tolist()
        data['per_class_recall'] = per_class_recall.tolist()
        data['per_class_precision'] = per_class_precision.tolist()
        data['per_class_fscore'] = per_class_fscore.tolist()
        json.dump(data, f, indent=2)
        

def draw_average_confusion_matrix(result_paths, img_path) -> None:
    plot_data_dict = defaultdict(list)
    for result_path in result_paths:
        with open(result_path, 'r') as file:
            data = json.load(file)
            confusion_matrix = data['confusion_matrix']
            labels = data['classes']
            N = len(labels)
            for i in range(N):
                for j in range(N):
                    plot_data_dict[(labels[i], labels[j])] += [confusion_matrix[i][j]]
    
    plot_data_dict_avg = { key : np.average(plot_data_dict[key]) for key in plot_data_dict }
    classes = sorted(list(set([ key[0] for key in plot_data_dict ] + [ key[1] for key in plot_data_dict ])))
    N = len(classes)
    avg_confusion_matrix = np.array([ [ plot_data_dict_avg[(classes[i],classes[j])] for i in range(N)] for j in range(N)])
    
    default_rcParams = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (20,20)
    ConfusionMatrixDisplay(avg_confusion_matrix, display_labels=classes).plot()
    plt.savefig(img_path)
    plt.rcParams["figure.figsize"] = default_rcParams


def draw_per_class_metric(metric, result_paths, img_path) -> None:
    plot_data_dict = defaultdict(list)
    for result_path in result_paths:
        with open(result_path, 'r') as file:
            data = json.load(file)
            labels = data['classes']
            per_class_metric = data[f'per_class_{metric}']
            for label, metric_value in zip(labels, per_class_metric):
                plot_data_dict[label] += [metric_value]
    classes = []
    plot_data = []
    for label in plot_data_dict:
        classes += [label]
        plot_data += [plot_data_dict[label]]

    _, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax.set_xticklabels(classes)
    ax.boxplot(plot_data)
    plt.title(f'{len(result_paths)}-fold per class {metric}')
    plt.savefig(img_path)


def draw_average_precision_recall_fscore(result_paths, img_path) -> None:
    precision = []
    recall = []
    f1score = []
    f2score = []
    f05score = []

    for result_path in result_paths:
        with open(result_path, 'r') as file:
            data = json.load(file)
            precision += [data['precision']]
            recall += [data['recall']]
            f1score += [data['f1score']]
            f2score += [data['f2score']]
            f05score += [data['f0.5score']]
    plot_data = [precision, recall, f1score, f2score, f05score]

    _, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax.set_xticklabels(['Precision', 'Recall', 'F1 Score', 'F2 Score', 'F0.5 Score'])
    ax.boxplot(plot_data)
    plt.title(f'{len(result_paths)}-fold average metrics')
    plt.savefig(img_path)


def kfold_validation_results(dir) -> None:
    filepaths = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.json') ]
    draw_average_precision_recall_fscore(filepaths, os.path.join(dir, 'average_results.jpeg'))
    draw_per_class_metric('precision', filepaths, os.path.join(dir, 'per_class_precision.jpeg'))
    draw_per_class_metric('recall', filepaths, os.path.join(dir, 'per_class_recall.jpeg'))
    draw_per_class_metric('fscore', filepaths, os.path.join(dir, 'per_class_fscore.jpeg'))
    draw_average_confusion_matrix(filepaths, os.path.join(dir, 'confusion_matrix.jpeg'))