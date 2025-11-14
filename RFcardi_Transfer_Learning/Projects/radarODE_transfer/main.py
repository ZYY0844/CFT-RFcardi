import torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = ((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)
print(BASE_DIR)
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.utils import set_random_seed, set_device
from LibMTL.model import resnet_dilated
from LibMTL import Trainer
from Projects.radarODE_transfer.utils.utils import shapeMetric, shapeLoss, ppiMetric, ppiLoss, anchorMetric, anchorLoss

from spectrum_dataset import dataset_concat
from nets.PPI_decoder import PPI_decoder
from nets.anchor_decoder import anchor_decoder
from nets.model import backbone, shapeDecoder
from config import prepare_args
import argparse



def parse_args(parser):
    parser.add_argument('--train_bs', default=32, type=int,
                        help='batch size for training')
    parser.add_argument('--test_bs', default=32, type=int,
                        help='batch size for test')
    parser.add_argument('--epochs', default=200,
                        type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/',
                        type=str, help='dataset path')
    # if True, only select 100 samples for training and testing
    parser.add_argument('--select_sample', default=False,
                        type=bool, help='select sample')
    parser.add_argument('--aug_snr', default=100, type=int, help='100 for no aug otherwise the SNR')
    return parser.parse_args()


def main(params, hyper,portion):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    id = int(portion/100*80)
    trian_id = id

    ID_all = np.arange(1, trian_id)
    ID_test = np.array([1,2,3,4,5,6,7,8,9,10])
    ID_train = np.delete(ID_all, ID_test-1)
    print('ID_test', ID_test,"ID_train", trian_id)

    # witout data augmentation
    radarODE_train_set = dataset_concat(
        ID_selected=ID_train, data_root=params.dataset_path, aug_snr=params.aug_snr)
    radarODE_test_set = dataset_concat(
        ID_selected=ID_test, data_root=params.dataset_path)


    trainloader = torch.utils.data.DataLoader(
        dataset=radarODE_train_set, batch_size=params.train_bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        dataset=radarODE_test_set, batch_size=params.test_bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # define tasks
    task_dict = {
                # for fine-tuning, comment out during pre-training
                'ECG_shape': {'metrics': ['norm_MSE', 'MSE', 'CE'],
                               'metrics_fn': shapeMetric(),
                               'loss_fn': shapeLoss(),
                               'weight': [0, 0, 0]},    
                # # for SSL stage, comment out during fine-tuning                    
                #  'Anchor': {'metrics': ["mse", 'sparseness'],
                #             'metrics_fn': anchorMetric(),
                #             'loss_fn': anchorLoss(),
                #             'weight': [0]}
                            }

    # # define backbone and en/decoders
    backbone_out_channels = hyper
    def encoder_class(): 
        return backbone(in_channels=10, out_channels=backbone_out_channels)
    
    num_out_channels = {'PPI': 260, 'Anchor': 800}
    decoders = nn.ModuleDict({'ECG_shape': shapeDecoder(in_channels=backbone_out_channels),
                              'Anchor': anchor_decoder(dim=backbone_out_channels)
                              })

    class radarODE_plus(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, modelName, aug_snr, **kwargs):
            super(radarODE_plus, self).__init__(task_dict=task_dict,
                                             weighting=weighting,
                                             architecture=architecture,
                                             encoder_class=encoder_class,
                                             decoders=decoders,
                                             rep_grad=rep_grad,
                                             multi_input=multi_input,
                                             optim_param=optim_param,
                                             scheduler_param=scheduler_param,
                                             modelName=modelName,
                                             aug_snr=params.aug_snr,
                                             **kwargs)


    radarODE_plus_model = radarODE_plus(task_dict=task_dict,
                          weighting=params.weighting,
                          architecture=params.arch,
                          encoder_class=encoder_class,
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          modelName=params.save_name,
                          aug_snr=params.aug_snr,
                          **kwargs)
    if params.mode == 'train':
        radarODE_plus_model.train(trainloader, testloader, params.epochs)
    elif params.mode == 'test':
        radarODE_plus_model.test(testloader)
    else:
        raise ValueError


if __name__ == "__main__":
    n_epochs = 200
    batch_size = 8
    learning_rate = 5e-3
    lr_scheduler = 'cos'
    optimizer = 'sgd'
    weight_decay=5e-4  
    momentum=0.937
    eta_min=learning_rate * 0.01 
    T_max=100
    params = parse_args(LibMTL_args)
    params.gpu_id = '0'
    dataset_select = 'RFcardi'
    params.dataset_path = f'/home/zhangyuanyuan/Dataset/data_{dataset_select}/'
    params.save_path = '/home/zhangyuanyuan/radarODE-Transfer/Model_saved/'
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    params.train_bs, params.test_bs = batch_size, batch_size
    params.epochs = n_epochs
    params.weighting = 'EW'
    params.rep_grad = False
    params.multi_input = False
    params.arch = 'HPS'
    params.optim = optimizer
    params.lr, params.weight_decay, params.momentum = learning_rate, weight_decay, momentum
    params.scheduler = lr_scheduler
    params.eta_min, params.T_max = eta_min, T_max
    params.mode = 'train'
    channel = 512
    portion = 100 # how many percentage of dataset for training
    # params.save_name = f'SSL_{portion}_'
    params.save_name = f'Super_{portion}_mse'
    # params.load_path= f'/home/zhangyuanyuan/radarODE-Transfer/Model_saved/best_SSL_100_spar.pt' # load pre-trained model if needed
    # params.save_name = f'FS_40_{method}_tt' # 100-portion for the percentage of labelled data
    main(params,channel,portion)


