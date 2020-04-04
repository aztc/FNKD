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
    filters1, filters2, filters3 = filters
    conv_name_base = model_name+'_stage' + str(stage) 
    bn_name_base = conv_name_base
    
    x = Norm(input_tensor,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(block+0))
    x = Activation(x,name=bn_name_base + '_activation'+str(block+0),act_type='relu')
    x = Convolution(x,kernel=(1,1),num_filter=filters1,no_bias=True,name=conv_name_base + '_conv'+str(block+1))
    
    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(block+1))
    x = Activation(x,name=bn_name_base + '_activation'+str(block+1),act_type='relu')
    x = Convolution(x,kernel=(3,3),pad=dilate,no_bias=True,num_filter=filters2,name=conv_name_base + '_conv'+str(block+2),dilate=dilate)
    
    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(block+2))
    x = Activation(x,name=bn_name_base + '_activation'+str(block+2),act_type='relu')
    x = Convolution(x,kernel=(1,1),num_filter=filters3,no_bias=True,name=conv_name_base + '_conv'+str(block+3))
    
    x = x + input_tensor
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),model_name='resnet50_v20',dilate=(1,1)):
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
    filters1, filters2, filters3 = filters
  
    conv_name_base = model_name+'_stage' + str(stage) 
    bn_name_base = conv_name_base
    
    x = Norm(input_tensor,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(block+0))
    x_sc = Activation(x,name=bn_name_base + '_activation'+str(block+0),act_type='relu')
    x = Convolution(x_sc,kernel=(1,1),num_filter=filters1,no_bias=True,name=conv_name_base + '_conv'+str(block+0))
    
    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(block+1))
    x = Activation(x,name=bn_name_base + '_activation'+str(block+1),act_type='relu')
    x = Convolution(x,kernel=(kernel_size,kernel_size),stride=strides,pad=dilate,num_filter=filters2,
                    no_bias=True, dilate=dilate,name=conv_name_base + '_conv'+str(block+1))
    
    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=bn_name_base + '_batchnorm'+str(block+2))
    x = Activation(x,name=bn_name_base + '_activation'+str(block+2),act_type='relu')
    x = Convolution(x,kernel=(1,1),num_filter=filters3,no_bias=True,name=conv_name_base + '_conv'+str(block+2))
    
    shortcut = Convolution(x_sc,kernel=(1,1),stride=strides,num_filter=filters3,
                           no_bias=True,name=conv_name_base + '_conv'+str(block+3))

    x = x + shortcut
    return x




def ResNet50_V2(inputs,classes=1000,batch_size=144,model_name='resnetv20'):
    """Instantiates the ResNet50 architecture.
    """

    x = inputs[0]
    lam = inputs[1]
        
    x = Norm(x,fix_gamma=True,use_global_stats=False,eps=1e-5,name=model_name+'_batchnorm0')
    x = Convolution(x,
        num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3,3), no_bias=True, name=model_name+'_conv0')
    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=model_name+'_batchnorm1')
    x = Activation(x,name=model_name+'_relu0',act_type='relu')
    x = Pooling(x,kernel=(3,3),stride=(2,2),pad=(1,1),pool_type='max',name=model_name+'_pool0')
     
    x = conv_block(x, 3, [64, 64, 256], stage=1, block=0, strides=(1, 1),model_name=model_name)
    x = identity_block(x, 3, [64, 64, 256], stage=1, block=3,model_name=model_name)
    x = identity_block(x, 3, [64, 64, 256], stage=1, block=6,model_name=model_name)
    
    
    x_f = mx.sym.slice(x,begin=(None,None,None,None),end=(None,None,None,None),step=(-1,1,1,1))
    lam1 = mx.sym.slice(lam,begin=(0),end=(1))
    lam2 = mx.sym.slice(lam,begin=(1),end=(2))
    x_mix = mx.sym.broadcast_mul(x,lam1) + broadcast_mul(x_f,lam2)
    x = mx.sym.concat(x,x_mix,dim=0) 
    

    x = conv_block(x, 3, [128, 128, 512], stage=2, block=0,model_name=model_name)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block=3,model_name=model_name)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block=6,model_name=model_name)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block=9,model_name=model_name)
    
  
    
    x = conv_block(x, 3, [256, 256, 1024], stage=3, block=0,model_name=model_name)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block=3,model_name=model_name)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block=6,model_name=model_name)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block=9,model_name=model_name)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block=12,model_name=model_name)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block=15,model_name=model_name)
    
    x = conv_block(x, 3, [512, 512, 2048], stage=4, block=0,model_name=model_name)
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block=3,model_name=model_name)
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block=6,model_name=model_name)  
    
    x = Norm(x,fix_gamma=False,use_global_stats=False,eps=1e-5,name=model_name+'_batchnorm2')
    x = Activation(x,name=model_name+'_relu1',act_type='relu')
    
    x = Pooling(x,kernel=(3,3),stride=(2, 2),pool_type='avg',global_pool=True,name=model_name+'_pool1')
    x = Flatten(x,name=model_name+'_flatten0')
    
#    x = mx.sym.L2Normalization(x)
    
    weight = mx.sym.Variable(model_name+'_dense_weight')
#    weight = mx.sym.L2Normalization(weight)
    output = FullyConnected(x,num_hidden=classes,weight=weight,no_bias=True,name=model_name+'_dense1')
#   
    
    output1 = mx.sym.slice_axis(output,axis=0,begin=0,end=batch_size)
    mix_output = mx.sym.slice_axis(output,axis=0,begin=batch_size,end=2*batch_size)
    
    
    output1_f = mx.sym.slice(output1,begin=(None,None),end=(None,None),step=(-1,1))
    output_mix = mx.sym.broadcast_mul(output1,lam1) + broadcast_mul(output1_f,lam2)
    
#    x = mx.sym.L2Normalization(x)
#    weight = mx.sym.Variable(model_name+'_dense1_weight',shape=(classes,2048)) 
#    weight = mx.sym.L2Normalization(weight)
    
    
#    alpha_w = mx.sym.Variable(model_name+'_alpha',shape=(1,1),init=mx.init.Constant(3))
    
#    alpha_w = mx.sym.clip(alpha_w,2,6)
    
#    weight = mx.sym.broadcast_mul(weight, alpha_w)  
#    x = mx.sym.broadcast_mul(x, alpha_w+1) 
    
#    weight = weight * alpha
#    x = x * (alpha+2)
    
#    weight = mx.sym.expand_dims(weight,0)
#    x = mx.sym.expand_dims(x,1)
#    x = mx.sym.tile(x,(1,classes,1))
#    dis = mx.sym.broadcast_minus(x,weight)
#    x = -mx.sym.sum(dis*dis,axis=2)
#    x = x / alpha
    
    return output1,output_mix,mix_output



#inputs = mx.sym.Variable('data')
#lam = mx.sym.Variable('lam')
#outputs = ResNet50_V2([inputs,lam],200,batch_size=144)
#net = mx.gluon.SymbolBlock(outputs,[inputs,lam]) 
##import gluoncv
##gluoncv.utils.viz.plot_network(net,shape=)
#
#a = mx.viz.plot_network(outputs[1], shape={'data':(144,3,224,224),'lam':(2,)},save_format='png')
#a.view('12')