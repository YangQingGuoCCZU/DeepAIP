# DeepAIP
Deep Learning for Anti-inflammatory Peptide Prediction Using Pre-trained Protein Language Model Features Based on Contextual Self-Attention Network 

# 1. Requirements

Python >= 3.10.6

torch = 2.1.2

pandas = 2.1.4

scikit-learn = 11.0.2

ProtT5-XL-UniRef50 model,it can be downloaded at: https://huggingface.co/Rostlab/prot_t5_xl_uniref50


# 2. Description
we propose a context self-attention deep learning model, coupled with features extracted from a pre-trained protein language model, to predict Anti-inflammatory Peptides (AIP). The contextual self-attention module can effectively enhance and understand the features extracted from the pre-trained protein language model, resulting in high accuracy to predict AIP. Additionally, we compared the performance of features extracted from popular pre-trained protein language models available in the market. Finally, Prot-T5 features demonstrated the best comprehensive performance as the input features for our deep learning model named DeepAIP. 


# 3 Datasets

neg_555_test.fasta: this file contains 555 Non-AIPs used for model test

pos_342_test.fasta: this file contains 342 AIPs  used for model test

neg_2218_train.fasta:this file contains 2218 Non-AIPs used for model training

pos_1365_train.fasta:this file contains 1365 AIPs used for model training

# 4. How to Use

## 4.1 Set up environment for ProtTrans
1. Set ProtTrans follow procedure from https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main .

## 4.2 Extract features

1. Extract Prot-T5 feature: cd to the DeepAIP/code dictionary, 
and run "python3 prot-t5_feature_encoding.py",
the Prot-T5 feature will be extracted.

## 4.3 Train and test

1. cd to the DeepAIP/code dictionary,and run "python3 DeepAIP.py.py" for training and testing the model.

