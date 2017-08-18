import argparse
import find_mxnet
import mxnet as mx
import time
import os, sys
import logging
import importlib
sys.path.insert(0, "./settings")

def score(model, data_val, metrics, gpus, batch_size, rgb_mean, network,
          image_shape, data_nthreads, epoch, num_classes, scale=0.0167):
    # create data iterator
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    mean_max_pooling_size = tuple([int(i)/32-6 for i in image_shape.split(',')])[1:3]
    data = mx.io.ImageRecordIter(
        data_name          = 'data',
        label_name         = 'softmax_label',
        # ------------------------------------
        path_imgrec        = data_val,
        aug_seq            = 'aug_torch',
        label_width        = 1,
        data_shape         = data_shape,
        force2color        = True,
        preprocess_threads = data_nthreads,
        verbose            = True,
        # ------------------------------------ 
        batch_size         = batch_size,
        # ------------------------------------ 
        rand_mirror        = False,
        mean_r             = rgb_mean[0],
        mean_g             = rgb_mean[1],
        mean_b             = rgb_mean[2],
        scale              = scale,
        # ------------------------------------  
        rand_crop          = False,
        min_random_area    = 1.0,
        max_random_area    = 1.0,
        fill_value         = (int(rgb_mean[0]),int(rgb_mean[1]),int(rgb_mean[2])), # TODO
        inter_method       = 2 # bicubic
        )

    # load parameters    
    prefix = model 
    _, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    logging.info('loading {}-{:04d}.params'.format(prefix, epoch))
    logging.info('loading {}-symbol.json'.format(prefix, epoch))
    
    # get before pooling
    sym  = importlib.import_module('symbol_' + network).get_before_pool()
    fc6_name = 'fc6-1k' if prefix.endswith('-extra') or prefix.endswith('extra-1k') else 'fc6'
    
    # set mean-max pooling
    sym  = mx.symbol.Pooling(data=sym, pool_type='avg', kernel=(7,7), stride=(1,1), pad=(0,0), name='avg_pool')
    sym  = mx.symbol.Convolution(data=sym, num_filter=num_classes, kernel=(1,1), no_bias=False, name=fc6_name)
    arg_params[fc6_name+'_weight'] = arg_params[fc6_name+'_weight'].reshape(arg_params[fc6_name+'_weight'].shape + (1,1))
    arg_params[fc6_name+'_bias']   = arg_params[fc6_name+'_bias'].reshape(arg_params[fc6_name+'_bias'].shape)
    sym1 = mx.symbol.Flatten(data=mx.symbol.Pooling(data=sym, pool_type='avg', kernel=mean_max_pooling_size, stride=(1,1), pad=(0,0), name='out_pool1'))
    sym2 = mx.symbol.Flatten(data=mx.symbol.Pooling(data=sym, pool_type='max', kernel=mean_max_pooling_size, stride=(1,1), pad=(0,0), name='out_pool2'))
    sym  = (sym1 + sym2 ) / 2.0
    sym  = mx.symbol.SoftmaxOutput(data = sym, name = 'softmax')
    
    if gpus == '':
        devs = mx.cpu()
    else:
        devs = [mx.gpu(int(i)) for i in gpus.split(',')]

    mod = mx.mod.Module(symbol=sym, context=devs)
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)
    if not isinstance(metrics, list):
        metrics = [metrics,]
    tic = time.time()
    num = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if num%10000==0:
            logging.info('num: {}: {}'.format(num,m.get()))
    return (num / (time.time() - tic), )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model',         type=str, required=True,)
    parser.add_argument('--gpus',          type=str, default='0,1')
    parser.add_argument('--batch-size',    type=int, default=200)
    parser.add_argument('--num-classes',   type=int, default=1000)
    parser.add_argument('--epoch',         type=int, default=0)
    parser.add_argument('--rgb-mean',      type=str, default='124,117,104')
    parser.add_argument('--data-val',      type=str, default='/tmp/val.rec')
    parser.add_argument('--image-shape',   type=str, default='3,320,320')
    parser.add_argument('--data-nthreads', type=int, default=8)
    parser.add_argument('--network',       type=str)
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k = 5)]

    (speed,) = score(metrics = metrics, **vars(args))
    logging.info('Finished with %f images per second', speed)

    for m in metrics:
        logging.info(m.get())
