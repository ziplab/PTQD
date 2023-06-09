import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

n_bits_w = 4
n_bits_a = 8

def calculate_t_channel_error(t_tensor, error_tensor):
    t_set = list(set(t_tensor.numpy().tolist()))
    bias_table=[[] for i in range(len(t_set))]
    channel_num = error_tensor.shape[1]
    for c in range(channel_num):
        channel_error_tensor = error_tensor[:,c,:,:]
        t_channel_error = torch.mean(channel_error_tensor, dim=(1,2))
        resultlist_list = [[] for _ in range(len(t_set))]
        for i in range(len(t_tensor)):
            resultlist_list[t_set.index(int(t_tensor[i]))].append(float(t_channel_error[i]))
        
        result_list = []
        for values in resultlist_list:
            result_list.append(sum(values)/len(values))
        
        idxs = t_set.copy()
        idxs.sort()
        for i in range(len(idxs)):
            idx=idxs[i]
            bias_table[i].append(result_list[t_set.index(idx)])
            
    bias_arr = np.array(bias_table)
    np.save('correct_data/imagenet_20steps_w{}a{}/idx_bias.npy'.format(n_bits_w, n_bits_a), bias_arr)
    print(bias_table)


def get_kt_dict_imagenet(data_tensor, error_tensor, t_tensor):

    t_data_dict, t_error_dict = {}, {}

    for i in range(len(t_tensor)):
        int_t = t_tensor[i].item()
        if int_t not in t_data_dict.keys():
            t_data_dict[int_t] = data_tensor[i]
            t_error_dict[int_t] = error_tensor[i]
        else:
            t_data_dict[int_t] = torch.cat([t_data_dict[int_t], data_tensor[i]], dim=0)
            t_error_dict[int_t] = torch.cat([t_error_dict[int_t], error_tensor[i]], dim=0)
    
    kt_dict = {}
    r_dict = {}
    for k in t_data_dict.keys():
        flatten_data = t_data_dict[k].flatten()
        flatten_error = t_error_dict[k].flatten()
        slope, intercept, r_value, p_value, std_err = st.linregress(flatten_data, flatten_error)
        
        kt_dict[k] = slope
        r_dict[k] = r_value
    
    # print('r_value: ', r_value)
    print(kt_dict)
    np.save('correct_data/imagenet_20steps_w{}a{}/kt.npy'.format(n_bits_w, n_bits_a), kt_dict, allow_pickle=True)

def get_t_residualerror_std_dict(t_tensor, data_tensor, error_tensor):
    '''
        need kt first for calculating residual error
    '''
    kt_dict = np.load('correct_data/imagenet_20steps_w{}a{}/kt.npy'.format(n_bits_w, n_bits_a), allow_pickle=True).item()
    t_std_dict = {}
    for i in range(len(t_tensor)):
        int_t = t_tensor[i].item()
        if int_t in kt_dict.keys():
            k = torch.tensor(kt_dict[int_t].astype('float32'))
            k = F.relu(k)
            k = k.item()
        else:
            k = 0
        residual_error = data_tensor[i] + error_tensor[i] - (1+k)*data_tensor[i]
        std = torch.std(residual_error)

        if int_t not in t_std_dict:
            t_std_dict[int_t] = [std]
        else:
            t_std_dict[int_t].append(std)
    for k in t_std_dict.keys():
        t_std_dict[k] = sum(t_std_dict[k]) / len(t_std_dict[k])
    print(t_std_dict)
    np.save("correct_data/imagenet_20steps_w{}a{}/t_std_dict.npy".format(n_bits_w, n_bits_a), t_std_dict, allow_pickle=True)


if __name__ == '__main__':
    data_error_t_list = torch.load('data_error_t_w{}a{}_scale3.0_eta0.0_step20.pth'.format(n_bits_w, n_bits_a), map_location='cpu')  ## replace error file here
    data_list = []
    error_list = [] 
    t_list = []
    for i in range(len(data_error_t_list)):
        for j in range(len(data_error_t_list[i][0])):
            data_list.append(data_error_t_list[i][0][j])
            error_list.append(torch.pow(data_error_t_list[i][1][j],1))
            t_list.append(data_error_t_list[i][2][j])

    data_tensor = torch.stack(data_list) 
    error_tensor = torch.stack(error_list)
    t_tensor = torch.stack(t_list)

    get_kt_dict_imagenet(data_tensor, error_tensor, t_tensor)
    calculate_t_channel_error(t_tensor, error_tensor)
    get_t_residualerror_std_dict(t_tensor, data_tensor, error_tensor)