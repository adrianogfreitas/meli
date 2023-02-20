from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np


def pair_plot(df: DataFrame, feature1: str, feature2: str, ax=None):
    met1_series = df[feature1]
    met2_series = df[feature2]

    corr = met1_series.corr(met2_series)

    title = f"({feature1}, {feature2}) Correlation = {corr:.2f}"

    if ax:
        ax.scatter(met1_series, met2_series, marker='.')
        ax.set_title(title)
        plt.grid()
    else:
        plt.scatter(met1_series, met2_series, marker='.')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(title)
        plt.tight_layout()
        plt.grid()

    
def correlation_matrix(df: DataFrame):
    corr_df = df.corr()
    ax = sns.heatmap(
        corr_df, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );


def plot_attributes(df: DataFrame, device: str):
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    df.loc[device][["attribute1"]].plot(ax=axs[0,0])
    axs[0,0].set_title("attribute1")
    df.loc[device][["attribute2"]].plot(ax=axs[0,1])
    axs[0,1].set_title("attribute2")
    df.loc[device][["attribute3"]].plot(ax=axs[0,2])
    axs[0,2].set_title("attribute3")
    df.loc[device][["attribute4"]].plot(ax=axs[1,0])
    axs[1,0].set_title("attribute4")
    df.loc[device][["attribute5"]].plot(ax=axs[1,1])
    axs[1,1].set_title("attribute5")
    df.loc[device][["attribute6"]].plot(ax=axs[1,2])
    axs[1,2].set_title("attribute6")
    df.loc[device][["attribute7"]].plot(ax=axs[2,0])
    axs[2,0].set_title("attribute7")
    df.loc[device][["attribute8"]].plot(ax=axs[2,1])
    axs[2,1].set_title("attribute8")
    df.loc[device][["attribute9"]].plot(ax=axs[2,2])
    axs[2,2].set_title("attribute9")

    for ax in axs.flat:
        ax.get_legend().remove()
        
    fig.tight_layout()


def pair_plot_attributes_target(df: DataFrame):
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    
    pair_plot(df, "failure", "attribute1", ax=axs[0,0])
    pair_plot(df, "failure", "attribute2", ax=axs[0,1])
    pair_plot(df, "failure", "attribute3", ax=axs[0,2])
    pair_plot(df, "failure", "attribute4", ax=axs[1,0])
    pair_plot(df, "failure", "attribute5", ax=axs[1,1])
    pair_plot(df, "failure", "attribute6", ax=axs[1,2])
    pair_plot(df, "failure", "attribute7", ax=axs[2,0])
    pair_plot(df, "failure", "attribute8", ax=axs[2,1])
    pair_plot(df, "failure", "attribute9", ax=axs[2,2])

    fig.tight_layout()


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15): 
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
    
    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """  
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes), # create enough axis slots for each class
           yticks=np.arange(n_classes), 
           xticklabels=labels, # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

def xgb_learning_curves(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    # plot log loss
    _, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()
    # plot classification error
    _, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()
