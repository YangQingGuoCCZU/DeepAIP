import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn.functional as F


#Model and Tokenizer Paths:
model_path = "D:/code/huggingface/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73"
tokenizer_path = model_path
#Loading the Dataset
df = pd.read_csv('../Dataset/test1.csv')
sequences = df['Sequence'].tolist()
sequences = [" ".join(seq) for seq in sequences]
label = df['label'].tolist()
label = np.array(label).reshape(-1, 1)
#Initializing the Tokenizer and Model
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
model = T5EncoderModel.from_pretrained(model_path)
#Tokenizing Sequences
max_length = 30
encoded_inputs = tokenizer(sequences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
#Getting Sequence Embeddings
with torch.no_grad():
    outputs = model(**encoded_inputs)
sequence_output = outputs.last_hidden_state
batch_size, seq_len, features = sequence_output.shape
sequence_output_reshaped = sequence_output.transpose(1, 2).reshape(batch_size * features, 1, seq_len)
kernel_size = 30
#Reshaping Sequence Output
pooling_output = F.avg_pool1d(sequence_output_reshaped, kernel_size=kernel_size, stride=kernel_size)
pooled_output_reshaped = pooling_output.reshape(batch_size, features, -1).mean(dim=2)
data = np.hstack((label, pooled_output_reshaped.numpy()))
features_df = pd.DataFrame(data, columns=['label'] + [f'Feature_{i+1}' for i in range(pooled_output_reshaped.shape[1])])
#Saving the Data
output_file = '../Dataset/test/prot-t5_test1.csv'
features_df.to_csv(output_file, index=False)
print('finish')


