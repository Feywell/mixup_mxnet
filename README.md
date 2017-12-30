# mixup_mxnet
This is a unofficial MXNet implementation of mixup method.

## Usage
1.simple example to train cifar10 
>python mixup_scratch.py

2.train cifar10 with multi gpus
>usage: mixup_scratch_cmd.py [-h] [--lr LR] [--batch_size BATCH_SIZE]  
>                            [--epochs EPOCHS] [--num_classes NUM_CLASSES]  
>                            [--use_mixup] [--alpha ALPHA]

>mxnet CIFAR10 Training

optional arguments:   
>  --h, --help                    show this help message and exit   
>  --lr LR                        learning rate                    
>  --batch_size BATCH_SIZE        batch size  
>  --epochs EPOCHS                train epoches  
>  --num_classes NUM_CLASSES      number of classes  
>  --use_mixup                    whether to use mixup or not  
>  --alpha ALPHA                  Beta distributed parmas   

*paper*: [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)

*Reference*:

- [1] [implementation by leehomyc with PyTorch](https://github.com/leehomyc/mixup_pytorch)
- [2] [MXNet image classification example (particularlly train_cifar10.py)](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification)
