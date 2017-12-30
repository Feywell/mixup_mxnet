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
import h5py
import os
import time
import argparse

parser = argparse.ArgumentParser(description='mxet CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epochs', default=200, type=int, help='train epoches')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--use_mixup', action='store_true',
                    help='whether to use mixup or not')
parser.add_argument('--alpha', default=1, type=float, help='Beta distributed parmas')


def shuffle_minibatch(inputs, targets, alpha=1,shapes=32,use_mixup=True,num_classes=10):
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
        a = np.random.beta(alpha, alpha, [batch_size, 1])
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

def evaluate(net, data_iter,batch_size,ctx=mx.gpu(0)):
    loss, acc = 0., 0.
    steps = 0.
    data_iter.reset()
    for i,batch in enumerate(data_iter):
        data, label = _get_batch(batch) 
        data = gluon.utils.split_and_load(data, ctx)
        label = gluon.utils.split_and_load(label, ctx)
#        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        losses = []
        outputs=[net(X) for X in data]
        losses = [softmax_cross_entropy(yhat, y) for yhat, y in zip(outputs, label)]
#        print output
        steps += batch_size
        acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
                          for yhat, y in zip(outputs, label)])
        loss += sum([l.sum().asscalar() for l in losses])
    return loss/steps, acc/steps

def _get_batch(batch):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (data, label)

def train(net,epochs,batch_size,data_iter_train,data_iter_val,use_mixup,num_classes,ctx):
    
    net.collect_params().reset_ctx(ctx)
    net.hybridize()    
    lr_sch = mx.lr_scheduler.MultiFactorScheduler(step=[38000,57000], factor=0.1)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',{
            'learning_rate':1e-1, 'momentum': 0.9,'wd':5e-4, 'lr_scheduler': lr_sch})
   
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        steps,m = 0.,0.
        data_iter_train.reset()
        tic = time.time() 
#        steps = len(data_iter_train)
        for i,batch in enumerate(data_iter_train):
            data, label = _get_batch(batch)

            losses = []
#            print label.shape
#            print label
            data, label = shuffle_minibatch(
                data,label,use_mixup=use_mixup,num_classes=num_classes)
#            print type(data)
#            print type(label)
#            print(data.shape,label.shape)
#            data, label = data.as_in_context(ctx), label.as_in_context(ctx)
            data = gluon.utils.split_and_load(data, ctx)
            label = gluon.utils.split_and_load(label, ctx)
#            print type(data[0])
#            print type(label[0])
#            print(data[0].shape,label[0].shape)
            with autograd.record():
#                print data
#                print type(data[0])
#                print data[0].shape
#                print data[0].shape
                outputs=[net(X) for X in data]
                losses = [-mx.nd.log_softmax(yhat) * y 
                          for yhat,y in zip(outputs,label) ]
                for l in losses:
                    l.backward()
#                output = net(data)
    #            print(output.shape,label.shape)
    #            print(label[0:2,:])
#                loss = -mx.nd.log_softmax(output) * label 
#                loss = loss.sum()/batch_size
    #            loss = softmax_cross_entropy(output, label)
    
#            loss.backward()
            trainer.step(batch_size)
            steps += batch_size
            m += sum([y.size for y in label])
#            print steps
            train_loss += sum([l.sum().asscalar() for l in losses])
            train_acc += sum([(yhat.argmax(axis=1)==y.argmax(axis=1)).
                              sum().asscalar()
                              for yhat, y in zip(outputs, label)])
            print(train_loss,steps)    
        val_loss,val_acc = evaluate(net,data_iter_val,batch_size,ctx)
        print("Epoch %d. loss: %.4f, acc: %.2f%%, val loss: %.4f, val acc: %.2f%%,time %.1f sec" 
              % (epoch+1, train_loss/steps, train_acc/m*100,val_loss,val_acc*100, time.time()-tic))
    
if __name__ == "__main__":
#    use_mixup = True
#    batch_size = 128
#    num_classes = 10
#    epochs = 200
    args = parser.parse_args()
    data_train,data_val = data_process(args.batch_size)

#    pretrained_net = model.vision.resnet18_v2(pretrained = True)
    finetune_net = model.vision.resnet18_v2(classes=args.num_classes)
#    finetune_net.features = pretrained_net.features
#    finetune_net.output.initialize(init.Xavier())
    ctx = []
    for i in get_gpus():
        ctx.append(mx.gpu(i))# 训练的时候为了简化计算，使用了单 GPU
    print ctx
    finetune_net.initialize(ctx=ctx,init=init.Xavier())
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()    
    
    train(finetune_net,args.epochs,args.batch_size,data_train,data_val,args.use_mixup,args.num_classes,ctx)
    val_loss,val_acc = evaluate(finetune_net, data_val,args.batch_size,ctx)
    print("val loss: %.4f, val acc: %.2f%%" % (val_loss, val_acc*100))