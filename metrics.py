import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, roc_curve

def plot_roc_curve(labels, output, one_hot=False, sample_weight=None):

    output_scores = []

    # Get ROC and AUC
    if(one_hot):
        output_scores = (np.array(output)[:,1]).tolist()
        #labels = np.array(labels).argmax(1).tolist()
    else:
        output_scores = output
    
    fpr, tpr, thresholds = roc_curve(labels, output_scores, sample_weight=sample_weight)
    roc_auc = roc_auc_score(labels, output_scores)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('ROC Test', fontsize=16, fontweight='bold')
    ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax.legend(loc='lower right', fontsize = 14)
    ax.plot([-0.1, 1.1], [-0.1, 1.1], 'r--')
    ax.set_xlabel('False Positive Rate', fontsize = 16)
    ax.set_ylabel('True Positive Rate', fontsize = 16)
    ax.axis([-0.01, 1.01, -0.01, 1.01])


def get_metrics(labels, outputs, threshold=0.5, target_names=None, one_hot=False):

    predicted_labels = []

    if(one_hot):
        output_scores = np.array(outputs).argmax(1).tolist()
        #labels = np.array(labels).argmax(1).tolist()
    else:
        output_scores = outputs
    predicted_labels = labels

    # Compute Confusion Matrix 
    cnf_matrix = confusion_matrix(labels, predicted_labels)
    report = classification_report(labels, predicted_labels, target_names=target_names)
    metrics = [report, cnf_matrix]

    return metrics
    

def get_roc_auc(labels, output, one_hot=False):

    # Get ROC and AUC
    if(one_hot):
        output_scores = np.array(output)[:,1].tolist()
        labels = np.array(labels).argmax(1).tolist()
    else:
        output_scores = output

    fpr, tpr, thresholds = roc_curve(labels, output_scores)
    roc_auc = roc_auc_score(labels, output_scores)
    metrics = [fpr, tpr, roc_auc]

    return metrics, thresholds


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()