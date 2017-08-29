from __future__ import print_function
import argparse

import matplotlib
import sys
import os
from time import time

import numpy

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers

import mlp

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from influence_functions import InfluenceFunctionsCalculator

def main(test_data_index=0):
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result_u100',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = mlp.MLP(args.unit, 10)
    classifier_model = L.Classifier(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make a specified GPU current
        classifier_model.to_gpu()  # Copy the model to the GPU

    # Load the model
    model_filepath = '{}/clf.model'.format(args.out)
    serializers.load_npz(model_filepath, classifier_model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    ifc = InfluenceFunctionsCalculator(classifier_model)
    ifc.calc_s_test(train, test[test_data_index:test_data_index+1])

    loss_list = []
    t0 = time()
    for index in range(len(train)):  #1000
        loss = ifc.I_up_loss(train[index:index+1])
        #print('index', index, loss)
        loss_list.append(loss)
    t1 = time()
    print('time: {}'.format(t1 - t0))
    loss_array = numpy.asarray(loss_list)

    infl_index = numpy.argsort(loss_array).ravel()
    print('BEST100:  ', infl_index[:30], loss_array[infl_index[:30]])
    print('WORST100: ', infl_index[::-1][:30], loss_array[infl_index[::-1][:30]])

    # Visualize test image
    if not os.path.exists(str(test_data_index)):
        os.mkdir(str(test_data_index))

    plt.figure()
    img = (test[test_data_index][0] * 255).astype(numpy.int32).reshape(28, 28)
    plt.imshow(img, cmap='gray')
    #plt.title("No.{0} / Answer:{1}, Predict:{}".format(i, test[i][1], prediction))
    plt.title(
        "No.{0} / Answer:{1}".format(test_data_index, test[test_data_index][1]))
    plt.savefig('{}/test.png'.format(test_data_index))

    def plot(mode='best'):
        ROW = 4
        COLUMN = 5
        # show graphical results of first 20 data to understand what's going on in inference stage
        plt.figure(figsize=(15, 10))
        for i in range(ROW * COLUMN):
            # Example of predicting the test input one by one.
            if mode == 'best':
                ind = infl_index[i]
            elif mode == 'worst':
                ind = infl_index[::-1][i]
            else:
                raise ValueError('mode must be either best or worst')
            print('i={}, infl_index={}'.format(i, ind))
            xp = chainer.cuda.cupy if args.gpu >= 0 else numpy
            x = xp.asarray([train[ind][0]])  # test data
            # t = Variable(xp.asarray([test[i][1]]))  # labels
            y = model(x)
            numpy.set_printoptions(precision=2, suppress=True)
            print('{}-th image: answer = {}, predict = {}'.format(i, train[ind][1], F.softmax(y).data))
            prediction = y.data.argmax(axis=1)
            example = (train[ind][0] * 255).astype(numpy.int32).reshape(28, 28)
            plt.subplot(ROW, COLUMN, i+1)
            plt.imshow(example, cmap='gray')
            plt.title("{}/A{},P:{},I={:.2}"
                      .format(i, train[ind][1], prediction, loss_array[ind]))
            plt.axis("off")
        plt.tight_layout()
        plt.savefig('{}/train_{}.png'.format(test_data_index, mode))

    plot('best')
    plot('worst')


if __name__ == '__main__':
    #for i in range(30):
    #    main(test_data_index=i)
    main(test_data_index=29)
