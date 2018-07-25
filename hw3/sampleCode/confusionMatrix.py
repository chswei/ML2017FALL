from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import *
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    '''
    This function prints and plots the confusion matrix.
    '''
    # 算出正確的比例，參考https://www.jianshu.com/p/5fe104e0fa70中confusion_matrix的功用
    ## np.newaxis: https://blog.csdn.net/qq1483661204/article/details/73135766
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot cm
    ## interpolation: https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html?highlight=interpolation
    ## cmap stands for colormap 決定圖的顏色 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    ## itertools.product: http://wiki.jikexueyuan.com/project/explore-python/Standard-Modules/itertools.html
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    emotion_classifier = load_model('model/model1-00050-0.66469.h5')
    ## https://blog.csdn.net/GZHermit/article/details/72716619
    np.set_printoptions(precision=2)
    with open('pkl/test_with_ans_pixels.pkl', 'rb') as f:
        X = pickle.load(f)
    predictions = emotion_classifier.predict(X)
    ## https://stackoverflow.com/questions/47435526/what-is-the-meaning-of-axis-1-in-keras-argmax
    predictions = predictions.argmax(axis=-1)
    print(predictions)
    with open('pkl/test_with_ans_labels.pkl', 'rb') as f:
        y = pickle.load(f)
    print(y)
    # 下面這句是重點
    ## https://www.jianshu.com/p/5fe104e0fa70
    conf_mat = confusion_matrix(y, predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
    plt.show()

if __name__ == '__main__':
    main()
