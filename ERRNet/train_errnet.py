from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data


def parse_csv_numbers(value, cast=float):
    return [cast(item.strip()) for item in value.split(',') if item.strip()]


def parse_ratios(value, expected_count):
    ratios = parse_csv_numbers(value)
    if len(ratios) != expected_count or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError('expected {} sampling ratios summing to 1, got {}'.format(expected_count, ratios))
    return ratios


opt = TrainOptions().parse()

cudnn.benchmark = True

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
datadir = './datasets/processed_data'

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True, 
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma,
    kernel_sizes=parse_csv_numbers(opt.synthetic_kernel_sizes, int),
    low_alpha=opt.low_alpha, high_alpha=opt.high_alpha,
    ghost_probability=opt.ghost_probability, ghost_max_shift=opt.ghost_max_shift,
    ghost_strength=opt.ghost_strength, random_reflection_pair=opt.random_reflection_pair)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_fusion = datasets.FusionDataset(
    [train_dataset, train_dataset_real],
    parse_ratios(opt.aligned_fusion_ratios, 2))

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=not opt.no_pin_memory)

eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))

eval_dataset_real = datasets.CEILTestDataset(
    join(datadir, 'real20'),
    size=20,
    max_long_edge=512)

eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=not opt.no_pin_memory)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=not opt.no_pin_memory)


"""Main Loop"""
engine = Engine(opt)

def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

if opt.resume:
    res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')

# define training strategy 
gan_weight = engine.model.opt.lambda_gan
if opt.gan_start_epoch < 0 or engine.epoch < opt.gan_start_epoch:
    engine.model.opt.lambda_gan = 0
set_learning_rate(opt.lr)
target_epoch = engine.epoch + opt.extra_epochs if opt.extra_epochs is not None else opt.nEpochs
while engine.epoch < target_epoch:
    if engine.epoch == opt.gan_start_epoch:
        engine.model.opt.lambda_gan = gan_weight
    if engine.epoch == 30:
        set_learning_rate(5e-5)
    if engine.epoch == 40:
        set_learning_rate(1e-5)
    if engine.epoch == opt.late_aligned_fusion_epoch:
        ratio = parse_ratios(opt.late_aligned_fusion_ratios, 2)
        print('[i] adjust fusion ratio to {}'.format(ratio))
        train_dataset_fusion.fusion_ratios = ratio
        set_learning_rate(5e-5)
    if engine.epoch == 50:
        set_learning_rate(1e-5)

    engine.train(train_dataloader_fusion)
    
    if engine.epoch % opt.eval_freq == 0:
        engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')        
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20')
