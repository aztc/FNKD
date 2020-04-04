# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from __future__ import print_function
from __future__ import absolute_import


from mxnet.symbol import Convolution,Variable,Activation,BatchNorm,Reshape,Flatten
from mxnet.symbol import FullyConnected,Pooling,Pooling_v1,flatten,SoftmaxOutput
from mxnet.symbol import broadcast_mul,UpSampling
from mxnet.symbol.contrib import BilinearResize2D
import mxnet as mx
from mxnet import nd
import numpy as np




def Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=None):
    running_mean = mx.sym.Variable(name=name+'_running_mean')
    running_var = mx.sym.Variable(name=name+'_running_var')
    x = BatchNorm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,
                  moving_mean=running_mean,moving_var=running_var,name=name)
    return x

def SE_block(x,nb_filter,block,stage):
    y = Pooling_v1(x,name='conv'+str(stage)+'_'+str(block)+'_global_pool',kernel=(2,2),pool_type='avg',global_pool=True)
    y = FullyConnected(y, name='conv'+str(stage)+'_'+str(block)+'_1x1_down', num_hidden=int(nb_filter/16))
    y = Activation(y, name='conv'+str(stage)+'_'+str(block)+'_1x1_down_relu', act_type='relu')
    y = FullyConnected(y, name='conv'+str(stage)+'_'+str(block)+'_1x1_up', num_hidden=nb_filter)
    y = Activation(y, name='conv'+str(stage)+'_'+str(block)+'_prob', act_type='sigmoid')
    y = broadcast_mul(Reshape(y,shape=(-1,nb_filter,1,1)),x)
    return y



def identity_block(input_tensor, kernel_size, filters, stage, block,model_name='resnet50_v20',dilate=(1,1)):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    conv_name_base = model_name+'_stage' + str(stage) 
    bn_name_base = conv_name_base
    
    x = Norm(input_tensor,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(2+0))
    x = Activation(x,name=bn_name_base + '_activation'+str(2+0),act_type='relu')
    x = Convolution(x,kernel=(3,3),pad=dilate,no_bias=True,num_filter=filters1,name=conv_name_base + '_conv'+str(block+0),dilate=dilate)
    
    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(2+1))
    x = Activation(x,name=bn_name_base + '_activation'+str(2+1),act_type='relu')
    x = Convolution(x,kernel=(3,3),pad=dilate,no_bias=True,num_filter=filters2,name=conv_name_base + '_conv'+str(block+1),dilate=dilate)
    
    x = x + input_tensor
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),short_connect=True,model_name='resnet50_v20',dilate=(1,1)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2 = filters
  
    conv_name_base = model_name+'_stage' + str(stage) 
    bn_name_base = conv_name_base
    
    x = Norm(input_tensor,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(block+0))
    x_sc = Activation(x,name=bn_name_base + '_activation'+str(block+0),act_type='relu')
    x = Convolution(x_sc,kernel=(kernel_size,kernel_size),stride=strides,pad=dilate,num_filter=filters1,
                    no_bias=True,dilate=dilate,name=conv_name_base + '_conv'+str(block+0))
    
    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(block+1))
    x = Activation(x,name=bn_name_base + '_activation'+str(block+1),act_type='relu')
    x = Convolution(x,kernel=(kernel_size,kernel_size),stride=(1,1),pad=dilate,num_filter=filters2,
                    no_bias=True,dilate=dilate,name=conv_name_base + '_conv'+str(block+1))
    
    if short_connect == True:
        shortcut = Convolution(x_sc,kernel=(1,1),stride=strides,num_filter=filters2,
                               no_bias=True,name=conv_name_base + '_conv'+str(block+2))
    else:
        shortcut = input_tensor 

    x = x + shortcut
    return x





def ResNet18_V2(inputs,classes=1000,model_name='resnetv20'):
    """Instantiates the ResNet18 architecture.
    """
    x = inputs
    x = Norm(x,fix_gamma=True,use_global_stats=False,eps=1e-5,name=model_name+'_batchnorm0')
    x = Convolution(x,
        num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3,3), no_bias=True, name=model_name+'_conv0')
    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=model_name+'_batchnorm1')
    x = Activation(x,name=model_name+'_relu0',act_type='relu')
    x = Pooling(x,kernel=(3,3),stride=(2,2),pad=(1,1),pool_type='max',name=model_name+'_pool0')
     
    x = conv_block(x, 3, [64, 64], stage=1, block=0, strides=(1, 1),short_connect=False,model_name=model_name)
    x = identity_block(x, 3, [64, 64], stage=1, block=2,model_name=model_name)

    x = conv_block(x, 3, [128, 128], stage=2, block=0,model_name=model_name)
    x = identity_block(x, 3, [128, 128], stage=2, block=3,model_name=model_name)

    x = conv_block(x, 3, [256, 256], stage=3, block=0,model_name=model_name)
    x = identity_block(x, 3, [256, 256], stage=3, block=3,model_name=model_name)

    x = conv_block(x, 3, [512, 512], stage=4, block=0, model_name=model_name)
    x = identity_block(x, 3, [512, 512], stage=4, block=3, model_name=model_name)
    

    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=model_name+'_batchnorm2')
    x = Activation(x,name=model_name+'_relu1',act_type='relu')
    
    x = Pooling(x,kernel=(3,3),stride=(2, 2),pool_type='avg',global_pool=True,name=model_name+'_pool1')
    x = Flatten(x,name=model_name+'_flatten0')
    
    xl2 = mx.sym.norm(x,axis=1,keepdims=True)
    xl2 = mx.sym.clip(xl2,0,8)
    x1 = mx.sym.L2Normalization(x)
    x1 = mx.sym.broadcast_mul(x1,xl2)
    
    weight = mx.sym.Variable(model_name+'_dense1_weight')
    bias = mx.sym.Variable(model_name+'_dense1_bias')
    
    output = FullyConnected(x,num_hidden=classes,weight=weight,bias=bias,name=model_name+'_dense1')
    
#    weight1 =  mx.sym.L2Normalization(weight)
    output1 = FullyConnected(x1,num_hidden=classes,weight=weight,bias=bias,name=model_name+'_dense2')

    return output,output1


#


    