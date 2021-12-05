import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import pickle, os, warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from random import randint

import cv2
from torch.autograd import Function
from scipy.interpolate import interp1d

from linformer import Linformer
from vit_pytorch.efficient import ViT

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='vit invasive/non-invasive single model')

parser.add_argument("--invasive",type=int, default=1)
parser.add_argument("--multi",type=int, default=0)

parser.add_argument('--pred_lag', type=int, default=0)
parser.add_argument('--dim', type=int, default=0)
parser.add_argument('--mtx_width',type=int, default=0)
parser.add_argument('--patch_size',type=int, default=0)

parser.add_argument('--lin_depth',type=int, default=0) #12
parser.add_argument('--lin_heads',type=int, default=0) #8
parser.add_argument('--lin_k',type=int, default=0) #64
               

args = parser.parse_args()


test_auc_by_best_valid_auc_list = []

invasive = bool(args.invasive) # either True or False
multi = bool(args.multi) # either True or False

pred_lag = args.pred_lag # 300 for 5-min, 600 for 10-min, 900 for 15-min prediction or others
dim = args.dim
mtx_width = args.mtx_width
patch_size = args.patch_size
lin_depth = args.lin_depth
lin_heads = args.lin_heads
lin_k = args.lin_k


print ( 'Invasive: {} | Multi: {} | Pred lag: {}'.format ( invasive, multi, pred_lag ))
print( 'dim :{}  |   mtx_width : {}  |   patch_size : {}'.format(dim, mtx_width, patch_size))
print( 'Lin_depth : {}  |  Lin_heads : {}  |  Lin_k : {}'.format(lin_depth, lin_heads,lin_k))

for _ in range(3): #n번 반복 실험
    
    
    # Prespecifications

    task = 'classification' # either 'classification' or 'regression'
    #invasive = False # either True or False
    #multi = False # either True or False
    #pred_lag = args.pred_lag # 300 for 5-min, 600 for 10-min, 900 for 15-min prediction or others
    
    #mtx_width = 60  # 10*300
    mtx_height = 3000//mtx_width
    #patch_size = 10
    if multi == True:
        channels = 4 if invasive == True else 3
    else:
        channels = 1
    

    cuda_number = 0 # -1 for multi GPU support
    num_workers = 0
    batch_size = 128
    #max_epoch = 200
    max_epoch = 15

    train_ratio = 0.6 # Size for training dataset
    valid_ratio = 0.1 # Size for validation dataset
    test_ratio = 0.3 # Size for test dataset

    random_key = randint(0, 100000) # Prespecify seed number if needed
    #random_key = 15322

    dr_classification = 0.3 # Drop out ratio for classification model
    dr_regression = 0.0 # Drop out ratio for regression model

    csv_dir = './model/'+str(random_key)+'/csv/'
    pt_dir = './model/'+str(random_key)+'/pt/'

    if not ( os.path.isdir( csv_dir ) ):
        os.makedirs ( os.path.join ( csv_dir ) )

    if not ( os.path.isdir( pt_dir ) ):
        os.makedirs ( os.path.join ( pt_dir ) )


    # Establish dataset

    class dnn_dataset(torch.utils.data.Dataset):
        def __init__(self, abp, ecg, ple, co2, target, invasive, multi):
            self.invasive, self.multi = invasive, multi
            self.abp, self.ecg, self.ple, self.co2 = abp, ecg, ple, co2
            self.target = target

        def __getitem__(self, index):
            if self.invasive == True:
                if self.multi == True: # Invasive multi-channel model
                    return np.float32( np.vstack (( np.array ( self.abp[index] ),
                                                    np.array ( self.ecg[index] ),
                                                    np.array ( self.ple[index] ),
                                                    np.array ( self.co2[index] ) ) ) ), np.float32(self.target[index])
                else: # Invasive mono-channel model (arterial pressure-only model)
                    return np.float32( np.array ( self.abp[index] ) ), np.float32(self.target[index])       
            else:
                if self.multi == True: # Non-invasive multi-channel model
                    return np.float32( np.vstack (( np.array ( self.ecg[index] ),
                                                    np.array ( self.ple[index] ),
                                                    np.array ( self.co2[index] ) ) ) ), np.float32(self.target[index])
                else: # Non-invasive mono-channel model (photoplethysmography-only model)
                    return np.float32( np.array ( self.ple[index] ) ), np.float32(self.target[index])

        def __len__(self):
            return len(self.target)


    #################### Vision Transformer ##################



    efficient_transformer = Linformer(
        dim=dim,
        seq_len = (mtx_width//patch_size)*(mtx_height//patch_size) + 1,  
        depth=lin_depth,
        heads=lin_heads,
        k=lin_k
    )

    model = ViT(
        dim=dim,
        image_size= max(mtx_width,mtx_height),
        patch_size= patch_size, 
        num_classes=2,
        transformer=efficient_transformer,
        channels=channels, 
    )

    ###########################################################


    # Read dataset

    processed_dir = './processed/'

    file_list = np.char.split ( np.array ( os.listdir(processed_dir) ), '.' )
    case_list = []
    for caseid in file_list:
        case_list.append ( int ( caseid[0] ) )
    #print ( 'N of total cases: {}'.format ( len ( case_list ) ) )

    cases = {}
    cases['train'], cases['valid+test'] = train_test_split ( case_list,
                                                            test_size=(valid_ratio+test_ratio),
                                                            random_state=random_key )
    cases['valid'], cases['test'] = train_test_split ( cases['valid+test'],
                                                      test_size=(test_ratio/(valid_ratio+test_ratio)),
                                                      random_state=random_key )

#     for phase in [ 'train', 'valid', 'test' ]:
#         print ( "- N of {} cases: {}".format(phase, len(cases[phase])) )

    for idx, caseid in enumerate(case_list):
        filename = processed_dir + str ( caseid ) + '.pkl'
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            data['caseid'] = [ caseid ] * len ( data['abp'] )

            raw_records = raw_records.append ( pd.DataFrame ( data ) ) if idx > 0 else pd.DataFrame ( data )
    #########################################
    ############# nan 값 제거 ##############

    nan_list = set()
    for x in ['abp','ecg','ple','co2']:
        j = 0
        for i in raw_records[x]:
            if np.isnan(i).any() == True:
                nan_list.add(j)
            j += 1

    nan_list = list(nan_list)
    indexes_to_keep = set(range(raw_records.shape[0])) - set(nan_list)
    raw_records = raw_records.take(list(indexes_to_keep))

    #########################################
    raw_records = raw_records[(raw_records['map']>=20)&(raw_records['map']<=160)].reset_index(drop=True) # Exclude abnormal range


    # Define loader and model

    if task == 'classification':
        task_target = 'hypo'
        criterion = nn.BCEWithLogitsLoss()
    else:
        task_target = 'map'
        criterion = nn.MSELoss()

    #print ( '\n===== Task: {}, Seed: {} =====\n'.format ( task, random_key ) )
    #print ( 'Invasive: {}\nMulti: {}\nPred lag: {}\n'.format ( invasive, multi, pred_lag ))

    records = raw_records.loc[ ( raw_records['input_length']==30 ) &
                                ( raw_records['pred_lag']==pred_lag ) ]

    records = records [ records.columns.tolist()[-1:] + records.columns.tolist()[:-1] ]
    #print ( 'N of total records: {}'.format ( len ( records ) ))

    split_records = {}
    for phase in ['train', 'valid', 'test']:
        split_records[phase] = records[records['caseid'].isin(cases[phase])].reset_index(drop=True)
        #print ('- N of {} records: {}'.format ( phase, len ( split_records[phase] )))

    #print ( '' )

    ext = {}
    for phase in [ 'train', 'valid', 'test' ]:
        ext[phase] = {}
        for x in [ 'abp', 'ecg', 'ple', 'co2', 'hypo', 'map' ]:
            ext[phase][x] = split_records[phase][x]

    dataset, loader = {}, {}
    epoch_loss, epoch_auc = {}, {}

    for phase in [ 'train', 'valid', 'test' ]:
        
        #     # reshape 3000 ---> mtx_height * mtx_weight
        ext[phase]['abp'] = [i.reshape(mtx_height,mtx_width) for i in ext[phase]['abp']]
        ext[phase]['ecg'] = [i.reshape(mtx_height,mtx_width) for i in ext[phase]['ecg']]
        ext[phase]['ple'] = [i.reshape(mtx_height,mtx_width) for i in ext[phase]['ple']]
        ext[phase]['co2'] = [i.reshape(mtx_height,mtx_width) for i in ext[phase]['co2']]
        
        dataset[phase] = dnn_dataset ( ext[phase]['abp'],
                                        ext[phase]['ecg'],
                                        ext[phase]['ple'],
                                        ext[phase]['co2'],
                                        ext[phase][task_target],
                                        invasive = invasive, multi = multi )
        loader[phase] = torch.utils.data.DataLoader(dataset[phase],
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    shuffle = True if phase == 'train' else False )
        epoch_loss[phase], epoch_auc[phase] = [], []

    #Model development and validation

    torch.cuda.set_device(cuda_number)
    DNN = model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DNN = DNN.to(device)

    optimizer = torch.optim.Adam(DNN.parameters(), lr=0.0005)
    n_epochs = max_epoch

    best_loss, best_auc = 99999.99999, 0.0

    for epoch in range(n_epochs):

        target_stack, output_stack = {}, {}
        current_loss, current_auc = {}, {}
        for phase in [ 'train', 'valid', 'test' ]:
            target_stack[phase], output_stack[phase] =  [], []
            current_loss[phase], current_auc[phase] = 0.0, 0.0

        DNN.train()

        for dnn_inputs, dnn_target in loader['train']:

            dnn_inputs, dnn_target = dnn_inputs.to(device), dnn_target.to(device)
            optimizer.zero_grad()
            dnn_inputs = dnn_inputs.reshape(len(dnn_inputs),channels,mtx_height,mtx_width)
            dnn_output = DNN( dnn_inputs )
            loss = criterion(dnn_output.T[0], dnn_target)
            current_loss['train'] += loss.item()*dnn_inputs.size(0)

            loss.backward()
            optimizer.step()

        current_loss['train'] = current_loss['train']/len(loader['train'].dataset)
        epoch_loss['train'].append ( current_loss['train'] ) 

        for phase in [ 'valid', 'test']:

            DNN.eval()
            with torch.no_grad():
                for dnn_inputs, dnn_target in loader[phase]:

                    dnn_inputs, dnn_target = dnn_inputs.to(device), dnn_target.to(device)
                    dnn_inputs = dnn_inputs.reshape(len(dnn_inputs),channels,mtx_height,mtx_width)
                    dnn_output = DNN( dnn_inputs )
                    target_stack[phase].extend ( np.array ( dnn_target.cpu() ) )
                    output_stack[phase].extend ( np.array ( dnn_output.cpu().T[0] ) )
                    loss = criterion((dnn_output.T[0]), dnn_target)
                    current_loss[phase] += loss.item()*dnn_inputs.size(0)

                current_loss[phase] = current_loss[phase]/len(loader[phase].dataset)
                epoch_loss[phase].append ( current_loss[phase] ) 

        if task == 'classification':
            log_label = {}
            for phase in ['valid', 'test']:
                current_auc[phase] = roc_auc_score ( target_stack[phase], output_stack[phase] )
                epoch_auc[phase].append ( current_auc[phase] )
        else:
            reg_output, reg_target, reg_label = {}, {}, {}
            for phase in ['valid', 'test']:
                reg_output[phase] = np.array(output_stack[phase]).reshape(-1,1)
                reg_target[phase] = np.array(target_stack[phase]).reshape(-1,1)
                reg_label[phase] = np.where(reg_target[phase]<65, 1, 0)
                method = LogisticRegression(solver='liblinear')
                method.fit(reg_output[phase], reg_label[phase]) # Model fitting
                current_auc[phase] = roc_auc_score (reg_label[phase], method.predict_proba(reg_output[phase]).T[1])
                epoch_auc[phase].append ( current_auc[phase] )


        label_invasive = 'invasive' if invasive == True else 'noninvasive'
        label_multi = 'multi' if multi == True else 'mono'
        label_pred_lag = str ( int ( pred_lag / 60 ) ) + 'min'

        filename = task+'_'+label_invasive+'_'+label_multi+'_'+label_pred_lag

        pd.DataFrame ( { 'train_loss':epoch_loss['train'],
                            'valid_loss':epoch_loss['valid'],
                            'test_loss':epoch_loss['test'],
                            'valid_auc':epoch_auc['valid'],
                            'test_auc':epoch_auc['test'] } ).to_csv(csv_dir+filename+'.csv')

        best = ''
        if task == 'regression' and abs(current_loss['valid']) < abs(best_loss):
            best = '< ! >'
            last_saved_epoch = epoch
            best_loss = abs(current_loss['valid'])
            #torch.save(DNN.state_dict(), pt_dir+filename+'_epoch_best.pt' )
        elif task == 'classification' and abs(current_auc['valid']) > abs(best_auc):
            best = '< ! >'
            last_saved_epoch = epoch
            best_auc = abs(current_auc['valid'])
            test_auc_by_best_valid_auc = abs(current_auc['test'])
            #torch.save(DNN.state_dict(), pt_dir+filename+'_epoch_best.pt' )

        #torch.save(DNN.state_dict(),pt_dir+filename+'_epoch_{0:03d}.pt'.format(epoch+1) )

#         print ( 'Epoch [{:3d}] Train loss: {:.4f} / Valid loss: {:.4f} (AUC: {:.4f}) / Test loss: {:.4f} (AUC: {:.4f}) {}'.format
#                 ( epoch+1,
#                 current_loss['train'],
#                 current_loss['valid'], current_auc['valid'],
#                 current_loss['test'], current_auc['test'], best ) )
    #print("mean : {}  /  std : {}".format(np.mean(AUROC_list), np.std(AUROC_list)))
    #print("###  test_auc_by_best_valid_auc : {} ###".format(test_auc_by_best_valid_auc))
    test_auc_by_best_valid_auc_list.append(test_auc_by_best_valid_auc)


print("mean : {}  /  std : {}\n".format(np.mean(test_auc_by_best_valid_auc_list), np.std(test_auc_by_best_valid_auc_list)))
