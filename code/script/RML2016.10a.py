import random
import time
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import shutil
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, LinearLR
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from tllib.alignment.dann import ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from calflops import calculate_flops

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from plugins import eraser, utils
from dataset import dataset
from raw_model.get_model import get_model

import setproctitle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
proc_title = "zhang_yezhuo"
setproctitle.setproctitle(proc_title)

def main(args:argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    global era, cru
    cru = era.crumble[args.cru_name+"_"+args.time]
    print(args)
   
    '''
    DATA
    '''
    raw_dataset = dataset.RML2016_Dataset(hdf5_file=args.hdf5_file, samples_per_key=args.samples_per_key)
    split_rate = args.split_rate if args.inner_train else 1.0
    train_size = int(split_rate * len(raw_dataset))
    test_size = len(raw_dataset) - train_size
    args.iters_per_epoch = args.iters_per_epoch if args.manual_iter else int(train_size / args.batch_size)
    
    # train data 
    torch.manual_seed(args.seed_split)
    train_pack = [(random_split(raw_dataset, [train_size, test_size])[0], args.modulation_train), ]
    
    # test data
    if args.inner_train:
        torch.manual_seed(args.seed_split)
        test_pack = [(random_split(raw_dataset, [train_size, test_size])[1], args.modulation_test), ]
    else: 
        test_pack = list()
        for modulation in args.modulation_test:
            test_pack.append((dataset.TORCHSIG_Dataset_HDF5(hdf5_file=args.hdf5_file, snrs = args.snr_train, lengths=args.length_train,
                                                                    modulations = args.modulation_test, samples_per_key=args.samples_per_key, model=args.arch),modulation, ))
  
    # make dataloader
    for (data, *nargs) in train_pack:
        print("train args: {0}, number: {1}".format(nargs, len(data)))
    for (data, *nargs) in test_pack:
        print("test args: {0}, number: {1}".format(nargs, len(data)))
    train_loader = DataLoader(train_pack[0][0], batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader_pack = list()
    for (test_dataset, *nargs) in test_pack:
        test_loader_pack.append((DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True), nargs))
    train_source_iter = ForeverDataIterator(train_loader)
    train_pack = list()
    
    '''
    MODEL
    '''
    # create model
    print("=> using model '{}'".format(args.arch))
    _batch_size = args.batch_size
    args.batch_size = 1
    backbone = get_model(args)# baseline
    classifier = backbone.to(device)
    flops, macs, params = calculate_flops(model=classifier, input_shape=(1, 2, args.length_train), output_as_string=True, output_precision=4)
    print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    args.batch_size = _batch_size   
    backbone = get_model(args)# baseline
    classifier = backbone.to(device)

    # define optimizer and lr scheduler
    optimizer_warm = Adam(classifier.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer = Adam(classifier.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler_warm = LinearLR(optimizer_warm, start_factor=1/args.warm_epoch, end_factor=1, total_iters=args.warm_epoch)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (((epoch%4+1)*0.15)*((-1/100*epoch)+1)))
    
    # seed training
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    '''
    IF TEST, TEST ONLY
    '''
    if args.phase == 'test':
        for i, (test_loader, *nargs) in enumerate(test_loader_pack):
            acc1 = utils.validate(test_loader, classifier, args, device, tip="{1}".format(nargs))
            cru["{0}".format(nargs)]["test"] = acc1
            era.save(args.cru_name+"_"+args.time)
        return
        
    '''
    TRAIN
    '''
    # warming up
    for epoch in tqdm(range(args.warm_epoch), desc="WARMING UP  ", position=0):
        print("\nlr:", lr_scheduler_warm.get_last_lr()[0])
        # train for some epoch
        train(train_source_iter, classifier, optimizer_warm,
              lr_scheduler_warm, epoch, args, phase = "warm")
        # save latest checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        # evaluate on validation set
        for i, (test_loader, *nargs) in enumerate(test_loader_pack):
            acc1 = round(utils.validate(test_loader, classifier, args, device, tip="{0}".format(nargs)), 4)
            cru["{0}".format(nargs)][epoch] = acc1
            era.save(args.cru_name+"_"+args.time)
    
    # start training
    start=time.time()
    acc_list = [[0, ] for i in range(len(test_pack))]
    for epoch in tqdm(range(args.epochs), desc="TRAINING ", position=0):
        print("\nlr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, classifier, optimizer,
              lr_scheduler, epoch, args, phase = "train")
        # save latest checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        # evaluate on validation set
        for i, (test_loader, *nargs) in enumerate(test_loader_pack):
            acc1 = round(utils.validate(test_loader, classifier, args, device, tip="{0}".format(nargs)), 4)
            acc_list[i].append(acc1)
            cru["{0}".format(nargs)][epoch+args.warm_epoch] = acc1
            era.save(args.cru_name+"_"+args.time)
            if acc1 == max(acc_list[i]):
                shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            utils.validate(test_loader, classifier, args, device, tip="{0}".format(nargs))
            

    print("trainning time cost: {}".format(time.time()-start))
    for i, (_, *nargs) in enumerate(test_loader_pack):
        print("best_acc of {0}: {1}".format(nargs, max(acc_list[i])))
    logger.close()


def train(train_source_iter: ForeverDataIterator, model: ImageClassifier, optimizer: Adam,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, phase = "train"):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch)) if phase == "train" else ProgressMeter(
        args.warm_iter,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()

    end = time.time()
    start = time.time()
    for i in tqdm(range(args.iters_per_epoch if phase == "train" else args.warm_iter), desc="iteration", position=0):
        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device).float() if args.arch != "CVCNN" else x_s.to(device).type(torch.complex128)
        labels_s = labels_s.to(device)

        # measure data loading time
        tmp = time.time()
        data_time.update(tmp-end)

        # compute output
        y = model(x_s)
        tmp = time.time()-tmp
        batch_time.update(tmp)
        # loss calculation
        cls_loss = F.cross_entropy(y, labels_s)
        loss = cls_loss
        cls_acc = accuracy(y, labels_s)[0]
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        end = time.time()

        if i % args.print_freq == 0:
            entries = [progress.prefix + progress.batch_fmtstr.format(i)]
            entries += [str(meter) for meter in progress.meters]
            tqdm.write('\t'.join(entries))
    
    print("A epoch: {}".format(time.time()-start))
    lr_scheduler.step()
    global era, cru
    cru = era.crumble[(args.cru_name+"_"+args.time)]
    cru["loss"][epoch] = loss.item()
    era.save(args.cru_name+"_"+args.time)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baseline')
    # csv logger
    parser.add_argument("--era_name", type=str, default="baseline", help='name of the pkl file that saves all the logs.')    
    parser.add_argument("--cru_name", type=str, default="test", help='name of the csv file to be saved.')    
    parser.add_argument("--time", type=str, default=eraser.get_time(), help='start time of the program.')    
    
    # dataset parameters
    parser.add_argument('-dname', '--dataset_name', default="RML2016.10a", type=str,  help='dataset_name. ')
    parser.add_argument('--hdf5_file', default="/home/zhangyezhuo/modulation_attack/data/NEU_POWDER/POWDER_DATASET.hdf5", 
                        help='hdf5 dataset path. POWDER and TORCHSIG are supported')   
    
    parser.add_argument('-in', '--inner_train', type=bool, default=True, choices=[True, False],
                        help='whether train inside a setting. If true, use split rate.')  
    parser.add_argument('-spk', '--samples_per_key', default=None, help='samples_per_key. ')  
    parser.add_argument('--split_rate', type=float, default=0.6, help='training rate.')  
    parser.add_argument('--seed_split', type=int, default=114514, help='seed for dataset spliting.')  
    
    parser.add_argument('-snrtrain', '--snr_train', default=[5, ], help='snr for training.', nargs = "+")   
    parser.add_argument('-snrtest', '--snr_test', default=[5, ], help='snr for testing. ', nargs = "+")   
    
    parser.add_argument('-lentrain', '--length_train', default=1024, type=int, choices=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],  help='sample length for training. ')   
    parser.add_argument('-lentest', '--length_test', default=1024, type=int, choices=[128, 256, 512, 1024, 2048, 4096, 8192, 16384], help='standard for testing. SAME AS TRAIN.')   
    
    modulation_list = ["qam-16", "qam-32", "qam-64", "qam-256", "qam-1024", 
                 "qam_cross-32", "qam_cross-128", "qam_cross-512", 
                 "psk-2", "psk-4", "psk-8", "psk-16", "psk-32", "psk-64", 
                 "pam-4", "pam-8", "pam-16", "pam-32", "pam-64",
                 "ook", "ask-4", "ask-8", "ask-16", "ask-32", "ask-64",]
    MODS=['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    parser.add_argument('-modtrain', '--modulation_train', default=MODS, help='modulation for training. ', nargs = "+")   
    parser.add_argument('-modtest', '--modulation_test', default=MODS, help='modulation for testing. ', nargs = "+") 
    
    # model parameters
    parser.add_argument('-a', '--arch', default='MAMCA', help="raw model to be utilized.")
    
    # training parameters
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
    
    parser.add_argument('--warm_epoch', default=10, type=int, help='warm up epoch, be small')
    parser.add_argument('--warm_iter', default=10, type=int, help='Number of iterations per epoch for warming up, be small')
    
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-wd', '--weight_decay', default=0.002, type=float, metavar='WD', help='weight decay of optimizer')
    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-mi', '--manual_iter', default=False, type=bool, help='if you wanna manually set the iteration per epoch: args.iters-per-epoch')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int, help='manually set the number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=1000, type=int, metavar='N', help='print frequency (default: 1000)')
    parser.add_argument('--per-class-eval', action='store_true', default=True, help='whether output per-class accuracy during evaluation')
    
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('-l', "--log", type=str, help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('-ph', "--phase", type=str, default='train', choices=['train', 'test'], help="When phase is 'test', only test the model.")    
    args = parser.parse_args()
    
    # eraser读写
    args.class_names = args.modulation_train
    era=eraser.Eraser(args.log, name=args.era_name)
    if not os.path.exists(args.log+"{}.pkl".format(args.era_name)): era.save()
    era.load(args.log+"{}.pkl".format(args.era_name))
    era.add_crumble(args.cru_name+"_"+args.time)
    cru = era.crumble[(args.cru_name+"_"+args.time)]
    
    main(args)
