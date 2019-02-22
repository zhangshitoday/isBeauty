'''
Function:
	config
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''


# trainset - images dir.
imagespath = './SCUT-FBP5500_v2/Images'
# trainset - ground truth path.
labpath = 'SCUT-FBP5500_v2/All_Ratings.xlsx'
# image shape of network input.
img_shape = (224, 224)
# whether shuffle the trainset or not.
is_shuffle = True
# batch size while training and testing.
batch_size = 64
# the number of worker.
num_workers = 4
# whether use GPU or not while training and testing.
use_cuda = True
# assign the ids of gpu.
gpus = '0,1'
# the number of used gpu.
ngpus = 2
# the number of epoch while training.
num_epochs = 50
# save the model parameters every save_interval epoch.
save_interval = 5
# dir to save the model parameters.
backupdir = './weights'
# file to save log info while training and testing.
logfile = 'train.log'
# if the distance of pred and groundtruth is smaller than error_tolerance, we regard the pred as a right one.
error_tolerance = 0.5