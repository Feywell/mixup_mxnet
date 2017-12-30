# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 17:56:51 2017

@author: Administrator
"""

import mxnet as mx
from mxnet import init
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
#from mxnet import image
import pandas as pd
import numpy as np
from mxnet.gluon import nn
from mxnet.gluon import model_zoo as model
from mxnet.gluon.data import vision
from util import *
#import h5py
import os
import time

def shuffle_minibatch(inputs, targets, shapes=32,use_mixup=True,num_classes=10):
    """Shuffle a minibatch and do linear interpolation between images and labels.
    Args:
        inputs: a numpy array of images with size batch_size x H x W x 3.
        targets: a numpy array of labels with size batch_size x 1.
        mixup: a boolen as whether to do mixup or not. If mixup is True, we
            sample the weight from beta distribution using parameter alpha=1,
            beta=1. If mixup is False, we set the weight to be 1 and 0
            respectively for the randomly shuffled mini-batches.
    """
#    print inputs.shape
    batch_size = inputs.shape[0]
    rp1 =  np.random.permutation(batch_size)  ##生成随机置换
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.one_hot(num_classes)

    rp2 =  np.random.permutation(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]    
    targets2_2 = targets2.one_hot(num_classes)    

    if use_mixup is True:
        a = np.random.beta(1, 1, [batch_size, 1])
    else:
        a = np.ones((batch_size, 1))

    b = np.tile(a[..., None, None], [1,3,shapes,shapes])

    inputs1 = inputs1 * nd.array(b)
    inputs2 = inputs2 * nd.array(1 - b)

    c = np.tile(a, [1, num_classes])
    targets1_oh = targets1_1 * nd.array(c)
    targets2_oh = targets2_2 * nd.array(1 - c)

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh
#    print inputs_shuffle.shape
#    print targets_shuffle.shape
    
    return inputs_shuffle, targets_shuffle

def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(path)):
            os.makedirs(os.path.join(path))

def download_cifar10(data_dir):

    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"),
              os.path.join(data_dir, "cifar10_train.lst"),
              os.path.join(data_dir, "cifar10_val.lst")
                          )
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.lst', fnames[3])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.lst', fnames[2])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames

def data_process(batch_size):
    data_dir='data'
    mkdir_if_not_exist(data_dir)    
    
    """
    ### 网络条件好可直接使用gluon API
    mkdir_if_not_exist('data')
    train_ds = vision.CIFAR10(root='./data',train=True)
    valid_ds = vision.CIFAR10(root='./data',train=False)
    """
    train_ds,valid_ds,train_lst, val_lst= download_cifar10(data_dir)
#    print train_ds
#    print valid_ds
#    train_lst = os.path.join(data_dir, "cifar10_train.lst")
#    val_lst = os.path.join(data_dir, "cifar10_val.lst")
    train_data = mx.io.ImageRecordIter(
            path_imgrec=train_ds,
            path_imglist=train_lst,
            label_width = 1,
            data_shape=(3,32,32),
            pad=4,
            mean_r = 0.4914,
            mean_g = 0.4822,
            mean_b = 0.4465,
            std_r = 0.2023,
            std_g = 0.1994,
            std_b = 0.2010,
            data_name='data',
            label_name='label',
            batch_size=batch_size,
            shuffle=True,
            preprocess_threads  = 20,
            rand_crop           = True,
            rand_mirror         = True
    )
#    train_data.reset()
#    batch = train_data.next()
#    images = batch.data[0]
#    print batch
#    print batch.label
    valid_data = mx.io.ImageRecordIter(
            path_imgrec=valid_ds,
            path_imglist=val_lst,
            label_width = 1,
            data_shape=(3,32,32),
            pad=4,
            resize=32,
            mean_r = 0.4914,
            mean_g = 0.4822,
            mean_b = 0.4465,
            std_r = 0.2023,
            std_g = 0.1994,
            std_b = 0.2010,
            data_name='data',
            label_name='label',
            batch_size=batch_size,
            preprocess_threads  = 20,
            rand_crop           = False,
            rand_mirror         = False
            )
#    loader = gluon.data.DataLoader
#    train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
#    valid_data = loader(valid_ds, batch_size, shuffle=False, last_batch='keep')

    return train_data,valid_data

def accuracy(output, labels):
#    labels = labels.max(axis=1)
    return nd.mean(nd.argmax(output, axis=1) == nd.argmax(labels, axis=1)).asscalar()

def val_accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()

def evaluate(net, data_iter,ctx):
    loss, acc = 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)   
#        print output
        acc += val_accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss/steps, acc/steps

def train(net,epochs,batch_size,data_iter_train,num_classes,ctx):
    
    net.collect_params().reset_ctx(ctx)
    net.hybridize()    
    lr_sch = mx.lr_scheduler.FactorScheduler(step=1500, factor=0.5)
    trainer = gluon.Trainer(net.collect_params(), 'adam', 
                            {'learning_rate': 1e-3, 'lr_scheduler': lr_sch})
  
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        steps = 0
        data_iter_train.reset()
        tic = time.time()  
#        steps = len(data_iter_train)
        for i,batch in enumerate(data_iter_train):
            data = batch.data[0]
            label = batch.label[0]
#            print label.shape
#            print label
            data, label = shuffle_minibatch(
                data, label,num_classes=num_classes)
            data, label = data.as_in_context(ctx), label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
    #            print(output.shape,label.shape)
    #            print(label[0:2,:])
                loss = -mx.nd.log_softmax(output) * label 
                loss = loss.sum()/batch_size
    #            loss = softmax_cross_entropy(output, label)
    
            loss.backward()
            trainer.step(batch_size)
            steps += batch_size
            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)
            print(train_loss,steps)
        print("Epoch %d. loss: %.4f, acc: %.2f%%, time %.1f sec" % (epoch+1, 
                            train_loss/steps, train_acc/steps*100, time.time()-tic))
    
if __name__ == "__main__":
    use_mixup = True
    batch_size = 32
    num_classes = 10
    epochs = 10
    data_train,data_val = data_process(batch_size)

#    pretrained_net = model.vision.resnet18_v2(pretrained = True)
    finetune_net = model.vision.resnet18_v2(classes=num_classes)
#    finetune_net.features = pretrained_net.features
#    finetune_net.output.initialize(init.Xavier())
    
    ctx = mx.gpu() # 训练的时候为了简化计算，使用了单 GPU
    finetune_net.initialize(ctx=ctx,init=init.Xavier())
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()    
    
    train(finetune_net,epochs,batch_size,data_train,num_classes,ctx)
    val_loss,val_acc = evaluate(finetune_net, data_val,ctx)
    print("val loss: %.4f, val acc: %.2f%%" % (val_loss, val_acc*100))