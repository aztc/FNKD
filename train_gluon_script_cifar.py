import time, logging, os, math
import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo import vision
from mxnet import init
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler
from models.symbol_Resnet50_v2 import ResNet50_V2
from mxnet.image import RandomSizedCropAug
from mxnet.image import HorizontalFlipAug
from gluoncv.data import transforms as gcv_transforms




class Options():
    def __init__(self,**kwargs):
       self.data_dir             = 'data'
       self.rec_train            = None
       self.rec_val              = None
       self.batch_size           = 128
       self.dtype                = 'float32'
       self.num_gpus             = 1
       self.num_epochs           = 100
       self.lr                   = 0.1
       self.momentum             = 0.9
       self.wd                   = 5e-4
       self.lr_mode              = 'step'
       self.lr_decay             = 0.1
       self.lr_decay_period      = 0
       self.lr_decay_epoch       = '30,60,90'
       self.warmup_lr            = 0
       self.warmup_epochs        = 0
       self.input_size           = 224
       self.jitter_param         = 0.4
       self.lighting_param       = 0.1
       self.max_random_area      = 1
       self.min_random_area      = 0.36
       self.max_aspect_ratio     = 0.1
       self.max_rotate_angle     = 20
       self.num_workers          = 20
       self.mean_rgb             = [0,0,0]
       self.num_classes          = 1000
       self.num_examples         = 1281167
       self.no_wd                = True
       self.save_frequency       = 50
       self.save_dir             = 'weights'
       self.resume_epoch         = 0
       self.resume_params        = None
       self.resume_states        = None
       self.log_interval         = 50
       self.logging_file         = None
       self.mode                 = 'hybrid'
       self.mixup                = False
       self.mixup_alpha          = 0.2
       self.mixup_off_epoch      = 0
       self.attention            = False
       self.att_size             = 224
       self.use_pretrain         = False
       self.union                = True
       self.distill              = True
       self.temperature          = None
       self.student              = None
       self.tea_net              = None
       self.tea_net_params       = None
       self.att_net_params       = None
       self.model_name           = None
       self.alpha                = 0.5
       self.l2_weight            = 2
       self.use_rec              = True
       self.isact                = True
       self.norm                 = None
       self.init_params          = None
       self.norm_distill         = False
       self.norm_distill_w       = None
       
    def list_all_members(self):
        res = []
        for name,value in vars(self).items():
            res.append((name+' : '+str(value)))
        np.set_printoptions(threshold=2000)
        res = np.array(res)
        return res
       
opt = Options()



os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

opt.use_rec         = False
opt.data_dir        = r'data'
opt.rec_train       = 'birds-ori-train'
opt.rec_val         = 'birds-ori-test'
opt.batch_size      = 128
opt.num_gpus        = 1
opt.num_epochs      = 200
opt.num_classes     = 100
opt.num_examples    = 50000
opt.log_interval    = 100
opt.lr              = 0.1
opt.momentum        = 0.9
opt.wd              = 1e-4
opt.no_wd           = False
opt.lr_decay        = 0.1
opt.lr_decay_epoch  = '100,150'
opt.input_size      = 224
opt.logging_file    = 'logs/cifar100_resnet20_tean2_w9.log'
opt.save_frequency  = 50
opt.save_dir        = 'logs/cifar100_resnet20_tean2_w9'
opt.max_aspect_ratio = 0.1
opt.max_random_area  = 1
opt.min_random_area  = 0.25
opt.max_rotate_angle = 20
opt.jitter_param     = 0
opt.lighting_param   = 0
opt.mixup            = False 
opt.mixup_alpha      = 1
opt.attention        = False
opt.att_size         = 224
opt.use_pretrain     = False
opt.union            = False
opt.distill          = False
opt.temperature      = 3
opt.norm_distill     = True
opt.norm_distill_w   = 9
opt.alpha            = 1
opt.num_workers      = 0
opt.l2_weight        = 3
opt.isact            = True
opt.norm             = 2
opt.model_name = model_name = 'cifar_resnet20_v2'
 
if opt.norm is not None:
    opt.init_params = r'init_weights/0.8173-cifar_resnet56_v2-best-0106.params'
else:
    opt.init_params = r'init_weights/0.8173-cifar_resnet56_v2-best-0106.params'


opt.student          = 'resnetv20_1'        
opt.tea_net          = r'init_weights/0.8173-cifar_resnet56_v2-best-symbol.json'
opt.tea_net_params   = r'init_weights/0.8173-cifar_resnet56_v2-best-0106.params'
opt.att_net_params   = r'dogs_ori_dilat/0.8447-Resnet50_v2-best-0064.params'



filehandler = logging.FileHandler(opt.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(opt.list_all_members())

batch_size = opt.batch_size
classes = opt.num_classes
num_training_samples = opt.num_examples

num_gpus = opt.num_gpus

context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

lr_decay = opt.lr_decay
lr_decay_period = opt.lr_decay_period
if opt.lr_decay_period > 0:
    lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
else:
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
num_batches = num_training_samples // batch_size

lr_scheduler = LRSequential([
    LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
    LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                nepochs=opt.num_epochs - opt.warmup_epochs,
                iters_per_epoch=num_batches,
                step_epoch=lr_decay_epoch,
                step_factor=lr_decay, power=2)
])


#    kwargs = {'ctx': context, 'pretrained': True}
kwargs = {'ctx': context}

optimizer = 'SGD'
optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}


if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True


model_name = opt.model_name


if opt.use_pretrain:
    inputs = mx.sym.Variable('data')
    outputs = ResNet50_V2(inputs,classes=opt.num_classes)
    net = gluon.SymbolBlock(outputs,inputs)
    net.load_parameters(r'test_weights/0.8664-imagenet-Resnet50_v2-best-0072.params',ctx=context)
    
else:
    from models.symbol_Resnet_cifar import cifar_resnet20_v2
    kwargs = {'isact': opt.isact, 'norm': opt.norm,'classes':classes,
              'prefix':'cifarresnetv20_','switchout':2}
    net = cifar_resnet20_v2(**kwargs)
    net.initialize(ctx = context)
    net.collect_params().reset_ctx(context)
    
    
    

net.cast(opt.dtype)
if opt.resume_params is not None:
    net.load_parameters(opt.resume_params, ctx = context)

    
if opt.distill:
    from models.symbol_Resnet_cifar import cifar_resnet56_v2
    kwargs = {'isact':opt.isact, 'norm': opt.norm,'classes':classes,
              'prefix':'cifarresnetv56_','switchout':0}
    tea_net = cifar_resnet56_v2(**kwargs)
    tea_net.collect_params().load(opt.tea_net_params)
    tea_net.collect_params().reset_ctx(context)

    
elif opt.norm_distill:
    
    from models.symbol_Resnet_cifar import cifar_resnet56_v2
    kwargs = {'isact':opt.isact, 'norm': opt.norm,'classes':classes,
              'prefix':'cifarresnetv56_','switchout':1}
    tea_net = cifar_resnet56_v2(**kwargs)
    tea_net.collect_params().load(opt.tea_net_params)
    tea_net.collect_params().reset_ctx(context)

    
    
    
# Two functions for reading data from record file or raw images
    
def get_data_img(opt):
    
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label
    
    batch_size = opt.batch_size
    
    if opt.num_classes == 10:
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2023, 0.1994, 0.2010]
    elif opt.num_classes == 100:
        mean = [0.5070, 0.4865, 0.4409]
        std  = [0.2673, 0.2564, 0.2761]
    
    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    if opt.num_classes == 10:
        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(root='data/cifar10',train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', 
            num_workers=opt.num_workers)
    elif opt.num_classes == 100:
        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(root='data/cifar100',train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', 
            num_workers=opt.num_workers)
            
    
    if opt.num_classes == 10:
        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(root='data/cifar10',train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)
    elif opt.num_classes == 100:
        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(root='data/cifar100',train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=opt.num_workers)

    
    return train_data, val_data, batch_fn



def get_data_rec(opt):
    
    rec_train = os.path.join(opt.data_dir,opt.rec_train+'.rec')
    rec_train_idx = os.path.join(opt.data_dir,opt.rec_train+'.idx')
    rec_val = os.path.join(opt.data_dir,opt.rec_val+'.rec')
    rec_val_idx = os.path.join(opt.data_dir,opt.rec_val+'.idx')

    input_size = opt.input_size
    crop_ratio = 0.875
    resize = int(math.ceil(input_size / crop_ratio))

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label
    

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = opt.num_workers,
        shuffle             = True,
        batch_size          = opt.batch_size,

        data_shape          = (3, input_size, input_size),
        mean_r              = opt.mean_rgb[0],
        mean_g              = opt.mean_rgb[1],
        mean_b              = opt.mean_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = opt.max_aspect_ratio,
        max_random_area     = opt.max_random_area,
        min_random_area     = opt.min_random_area,
        max_rotate_angle    = opt.max_rotate_angle,
        brightness          = opt.jitter_param,
        saturation          = opt.jitter_param,
        contrast            = opt.jitter_param,
        pca_noise           = opt.lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = opt.num_workers,
        shuffle             = False,
        rand_crop           = False,
        batch_size          = opt.batch_size,
        resize              = resize,
        data_shape          = (3, input_size, input_size),
        mean_r              = opt.mean_rgb[0],
        mean_g              = opt.mean_rgb[1],
        mean_b              = opt.mean_rgb[2],
    )
    return train_data, val_data, batch_fn


if opt.use_rec:
    train_data, val_data, batch_fn = get_data_rec(opt)
else:
    train_data, val_data, batch_fn = get_data_img(opt)


if opt.mixup:
    train_metric = mx.metric.RMSE()
else:   
    train_metric = mx.metric.Accuracy()
    

acc_top1 = mx.metric.Accuracy()  


save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0



def mixup_transform(label, classes, lam=1, eta=0.0):
    if isinstance(label, nd.NDArray):
        label = [label]
    res = []
    for l in label:
        y1 = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
        y2 = l[::-1].one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
        res.append(lam*y1 + (1-lam)*y2)
    return res
    
    
def test(ctx, val_data, opt):
    
    if opt.use_rec:
        val_data.reset()
        val_data.reset() 
        
    acc_top1.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        
        if opt.norm_distill:
            outputs = [net(X.astype(opt.dtype, copy=False))[0] for X in data]
            acc_top1.update(label, outputs)
        else:
            outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            acc_top1.update(label, outputs)
            
            
    name, top1 = acc_top1.get()
    
    if type(name)==str:
        name = [name]
        top1 = [top1]
    return (name, top1)



ctx = context
if opt.mode == 'hybrid':
    net.hybridize(static_alloc=True, static_shape=True)

if isinstance(ctx, mx.Context):
    ctx = [ctx]
if opt.resume_params is '':
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

if opt.no_wd:
    for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
if opt.resume_states is not None:
    trainer.load_states(opt.resume_states)



if opt.mixup:
    sparse_label_loss = False
else:
    sparse_label_loss = True
      


if opt.distill:
    L_dis = gcv.loss.DistillationSoftmaxCrossEntropyLoss(temperature=opt.temperature,
                                                     hard_weight=0.5,
                                                     sparse_label=sparse_label_loss) 
if opt.norm_distill:
    Lndis = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

    
L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)


best_val_score = 0

for epoch in range(opt.resume_epoch, opt.num_epochs):
    
    tic = time.time()
    
    if opt.use_rec:
        train_data.reset()
        
    train_metric.reset()
    btic = time.time()
       
    for i, batch in enumerate(train_data):
        data, label = batch_fn(batch, ctx)


        if opt.distill:
            tea_outputs = [tea_net(X.astype(opt.dtype, copy=False)) for X in data]
            tea_prob = [nd.softmax(X/opt.temperature) for X in tea_outputs]
        elif opt.norm_distill:
            tea_outputs = [tea_net(X.astype(opt.dtype, copy=False)) for X in data] 
            tea_prob = [nd.softmax(X) for X in tea_outputs]
        
        with ag.record():
            

            if opt.norm_distill:
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                outputs_logits = [X[0] for X in outputs]  
                outputs_logits1 = [X[1] for X in outputs]    
            else:
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data] 
                
            if opt.distill:  
                loss = [L_dis(yhat, y.astype(opt.dtype, copy=False), p) for yhat, y, p in zip(outputs, label, tea_prob)]
            elif opt.norm_distill:
                l1 = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs_logits, label)]
                l2 = [Lndis(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs_logits1, tea_prob)]
                loss = [x+y*opt.norm_distill_w for x,y in zip(l1,l2)]  
            elif opt.union:
                l1 = [L(yhat[0], y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                l2 = [L(yhat[1], y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                loss = [x+y for x,y in zip(l1,l2)]
            else:
                loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()
        trainer.step(batch_size)

        if opt.mixup and opt.union:
            output_softmax = [nd.SoftmaxActivation(out[-1].astype('float32', copy=False)) \
                            for out in outputs]
            train_metric.update(label, output_softmax)
        
        elif opt.mixup:
            output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                            for out in outputs]
            train_metric.update(label, output_softmax)
        elif opt.norm_distill:
            train_metric.update(label, outputs_logits)
        else:
            train_metric.update(label, outputs)

        if opt.log_interval and not (i+1)%opt.log_interval:
            train_metric_name, train_metric_score = train_metric.get()
            logger.info('Epoch[%d] Batch [%d]  Speed: %.2f Hz  %s=%.4f  lr=%.4f'%(
                        epoch, i, batch_size*opt.log_interval/(time.time()-btic),
                        train_metric_name, train_metric_score, trainer.learning_rate))
            btic = time.time()

    train_metric_name, train_metric_score = train_metric.get()
    throughput = int(batch_size * i /(time.time() - tic))
    

    top1_name, top1_val = test(ctx, val_data, opt)  
    
    logger.info('Epoch[%d] Train-accuracy=%f'%(epoch, train_metric_score))
    logger.info('Epoch[%d] speed: %d samples/sec\ttime cost: %f'%(epoch, throughput, time.time()-tic))
    

    for idx in range(len(top1_name)):
        name = top1_name[idx]
        val  = top1_val[idx]
        logger.info('Epoch[%d] Validation-%s=%.4f', epoch, name, val)
    
    if len(top1_val) > 0:
        top1_val = top1_val[-1]
        
    if top1_val > best_val_score:
        best_val_score = top1_val
        net.export('%s/%.4f-%s-best'%(save_dir, best_val_score, model_name),epoch)

    if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
        net.export('%s/%s'%(save_dir, model_name),epoch)

if save_frequency and save_dir:
    net.export('%s/%s'%(save_dir, model_name),opt.num_epochs-1)

