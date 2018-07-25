import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored, cprint
from utils import *
from keras.models import load_model
import keras.backend as K

def deprocessimage(x):
    '''
    Hint: Normalize and Clip
    '''
    return x

## https://blog.csdn.net/ziyuzhao123/article/details/8811496
## __file__的值其實就是在shell命令列上invoke Python時給的script名稱
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir, 'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')

def main():
    parser = argparse.ArgumentParser(
        prog='saliencyMap.py', 
        description='ML-Assignment3 visualize attention heat map.'
    )
    parser.add_argument(
        '--epoch', 
        type=int, 
        metavar='<#epoch>',
        default=1
    )
    args = parser.parse_args()

    model_name = 'model-%s.h5' %str(args.epoch)
    model_path = os.path.join(model_dir, model_name)
    emotion_classifier = load_model(model_path)
    print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    with open('test_with_ans_pixels.pkl') as f:
        X = pickle.load(f)
    input_img = emotion_classifier.input
    img_ids = [17]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(X[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grad])

        '''
        Implement your heatmap processing here!
        Hint: Do some normalization or smoothening on grads
        '''

        thres = 0.5
        see = X[idx].reshape(48,48)
        # for i in range(48):
            # for j in range(48):
                # print(heatmap[i][j])
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.show()

        test_dir = os.path.join(cmap_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        
        test_dir = os.path.join(partial_see_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

if __name__ == '__main__':
    main()