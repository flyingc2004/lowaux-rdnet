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


def parse_lr_schedule(milestones_value, learning_rates_value):
    milestones = parse_csv_numbers(milestones_value, int)
    learning_rates = parse_csv_numbers(learning_rates_value)
    if len(milestones) != len(learning_rates):
        raise ValueError('unaligned LR milestones and values must have the same length')
    if milestones != sorted(set(milestones)) or any(epoch < 0 for epoch in milestones):
        raise ValueError('unaligned LR milestones must be unique non-negative epochs in ascending order')
    if any(lr <= 0 for lr in learning_rates):
        raise ValueError('unaligned LR values must be positive')
    return dict(zip(milestones, learning_rates))


opt = TrainOptions().parse()

cudnn.benchmark = True

# processed datasets prepared by datasets/prepare_train_data.py and datasets/prepare_test_data.py
datadir = './datasets/processed_data'
raw_datadir = './datasets/raw_data'

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')
datadir_unaligned = join(raw_datadir, 'Dataset/DSLR/unaligned_train250')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size,
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma,
    kernel_sizes=parse_csv_numbers(opt.synthetic_kernel_sizes, int),
    low_alpha=opt.low_alpha, high_alpha=opt.high_alpha,
    ghost_probability=opt.ghost_probability, ghost_max_shift=opt.ghost_max_shift,
    ghost_strength=opt.ghost_strength, random_reflection_pair=opt.random_reflection_pair)
train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_unaligned = datasets.CEILTestDataset(datadir_unaligned, enable_transforms=True, flag={'unaligned':True}, size=None)

train_dataset_fusion = datasets.FusionDataset(
    [train_dataset, train_dataset_unaligned, train_dataset_real],
    parse_ratios(opt.unaligned_fusion_ratios, 3))


train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=not opt.no_pin_memory)


engine = Engine(opt)
"""Main Loop"""
def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        util.set_opt_param(optimizer, 'lr', lr)


set_learning_rate(opt.lr)
lr_schedule = parse_lr_schedule(opt.unaligned_lr_milestones, opt.unaligned_lr_values)
if lr_schedule:
    print('[i] unaligned learning-rate schedule: {}'.format(lr_schedule))
target_epoch = engine.epoch + opt.extra_epochs if opt.extra_epochs is not None else 80
while engine.epoch < target_epoch:
    if engine.epoch in lr_schedule:
        set_learning_rate(lr_schedule[engine.epoch])
        
    engine.train(train_dataloader_fusion)
