import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def read_file_list(filename):
    f = open(filename)
    data = f.readlines()
    f.close()
    data=[idata.strip() for idata in data]
    return data

def read_file_list_from_seq_label(filename):
    f = open(filename)
    data = f.readlines()
    f.close()
    results=[]
    block=len(data)//2
    for index in range(block):
        item=data[index*2+0].split()
        name =item[0].strip()
        results.append(name)
    return results

def read_prob_label(filename):
    df = pd.read_csv(filename)
    probs = df[df.columns[1]].values.astype(np.float32)
    labels= df[df.columns[2]].values.astype(np.float32)
    return probs , labels

def save_prob_label(probs,labels,filename):
    #data={'probs':probs,'labels':labels}
    probs = np.array(probs)
    labels = np.array(labels)
    data = np.hstack((probs.reshape(-1, 1), labels.reshape(-1, 1)))
    names = ['probs', 'labels']
    Pd_data = pd.DataFrame(columns=names, data=data)
    Pd_data.to_csv(filename)


def create_src_lengths_mask(batch_size: int, src_lengths):

    max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()

def masked_softmax(scores, src_lengths, src_length_masking=True):
    #scores [batchsize,L*L]
    if src_length_masking:
        bsz, src_len,max_src_len = scores.size()
        # compute masks
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        src_mask = torch.unsqueeze(src_mask, 2)
        #print('scr_mask',src_mask)
        #scores=scores.permute(0,2,1)
        # Fill pad positions with -inf
        scores=scores.permute(0,2,1)
        scores = scores.masked_fill(src_mask == 0, -np.inf)
        scores = scores.permute(0, 2, 1)
        #print('scores',scores)
    return F.softmax(scores.float(), dim=-1)



