# model settings
model = 'resnet18'
# dataset settings
data_root = '/mnt/SSD/dataset/ILSVRC/Data/CLS-LOC'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
batch_size = 256

# optimizer and learning rate
optimizer = dict(
    algorithm='SGD', args=dict(lr=0.1, momentum=0.9, weight_decay=5e-4))
lr_policy = dict(policy='step', step=30)

# logging settings
log_level = 'INFO'
log_cfg = dict(
    # log at every 50 iterations
    interval=50,
    hooks=[
        ('TextLoggerHook', {}),
        # ('TensorboardLoggerHook', dict(log_dir=work_dir + '/log'))
    ])

# runtime settings
work_dir = './demo'
gpus = range(8)
data_workers = len(gpus) * 2
checkpoint_cfg = dict(interval=5)  # save checkpoint at every epoch
workflow = [('train', 5), ('val', 1)]
max_epoch = 90
