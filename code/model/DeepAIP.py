from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from LossFunction.focalLoss import FocalLoss_v2
from utility import save_prob_label, masked_softmax, read_file_list_from_seq_label

#This class implements a contextual attention mechanism,
# which combines local and global information from input embeddings
# using convolutional and attention operations.

#q_input_dim: Dimension of the query input
#v_input_dim: Dimension of the value input
# This corresponds to the size of the pre-trained model embeddings.
#qk_dim: Dimension of the query/key vectors in the attention mechanism
#v_dim: Dimension of the value vectors
class Contextual_Attention(nn.Module):
    def __init__(self, q_input_dim, v_input_dim=391, qk_dim=391, v_dim=391):
        super(Contextual_Attention, self).__init__()
        self.cn3 = nn.Conv1d(q_input_dim, qk_dim, 3, padding='same')
        self.cn5 = nn.Conv1d(q_input_dim, qk_dim, 5, padding='same')
        self.k = nn.Linear(v_dim * 2 + q_input_dim, qk_dim)
        self.q = nn.Linear(q_input_dim, qk_dim)
        self.v = nn.Linear(v_input_dim, v_dim)
        self._norm_fact = 1 / torch.sqrt(torch.tensor(qk_dim))
    def forward(self, plm_embedding, evo_local, seqlengths):
        Q = self.q(evo_local)
        k3 = self.cn3(evo_local.permute(0, 2, 1))
        k5 = self.cn5(evo_local.permute(0, 2, 1))
        evo_local = torch.cat((evo_local, k3.permute(0, 2, 1), k5.permute(0, 2, 1)), dim=2)
        K = self.k(evo_local)
        V = self.v(plm_embedding)
        atten = masked_softmax((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact, seqlengths)
        output = torch.bmm(atten, V)
        return output + V
#This function is used to prepare batches of training data for input into the model.
# Specifically, it handles padding sequences to ensure that all inputs in a batch have the same length.
def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature0 = []
    f0agv = []
    feature_fusion = []
    train_y = []
    for data in batch_traindata:
        feature0.append(data[0])
        f0agv.append(data[1])
        feature_fusion.append(data[2])
        train_y.append(data[3])
    data_length = [len(data) for data in feature0]
    mask = torch.full((len(batch_traindata), data_length[0]), False).bool()
    for mi, aci in zip(mask, data_length):
        mi[aci:] = True
    feature0 = torch.nn.utils.rnn.pad_sequence(feature0, batch_first=True, padding_value=0)
    f0agv = torch.nn.utils.rnn.pad_sequence(f0agv, batch_first=True, padding_value=0)
    feature_fusion = torch.nn.utils.rnn.pad_sequence(feature_fusion, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    return feature0, f0agv, feature_fusion, train_y, torch.tensor(data_length)
#This class is a custom dataset class for loading protein data and feature fusion data from CSV files for bioinformatics tasks.
#It inherits from torch.utils.data.Dataset and implements methods to load and process data.
class BioinformaticsDataset(Dataset):
    def __init__(self, X_prot, X_feature_fusion):
        self.X_prot = X_prot
        self.X_feature_fusion = X_feature_fusion
    def __getitem__(self, index):
        filename_prot = self.X_prot[index]
        df_prot = pd.read_csv('../DataSet/enhance/' + filename_prot)
        prot = df_prot.iloc[:, 1:].values
        if prot.dtype == object:
            prot = prot.astype(float)
        prot = torch.tensor(prot, dtype=torch.float)
        agv = torch.mean(prot, dim=0)
        agv = agv.repeat(prot.shape[0], 1)
        filename_feature_fusion = self.X_feature_fusion[index]
        df_feature_fusion = pd.read_csv('../DataSet/enhance/' + filename_feature_fusion)
        feature_fusion = df_feature_fusion.iloc[:, 1:].values
        feature_fusion = torch.tensor(feature_fusion, dtype=torch.float)
        label = df_prot.iloc[:, 0].values
        label = torch.tensor(label, dtype=torch.long)
        return prot, agv, feature_fusion, label
    def __len__(self):
        return len(self.X_prot)
#This class defines a neural network model for predicting anti-inflammatory peptides using protein sequences and feature fusion data.
#It leverages a contextual attention mechanism (Contextual_Attention class) and convolutional layers for feature extraction.
class DeepAIPModule(nn.Module):
    def __init__(self):
        self.ca = Contextual_Attention(q_input_dim=391, v_input_dim=391)
        self.relu = nn.ReLU(True)
        self.protcnn1 = nn.Conv1d(391 + 391 + 391, 512, 3, padding='same')
        self.protcnn2 = nn.Conv1d(512, 256, 3, padding='same')
        self.protcnn3 = nn.Conv1d(256, 128, 3, padding='same')
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 2)
        self.drop = nn.Dropout(0.5)
    def forward(self, prot0, f0agv, evo, data_length):
        evosa = self.ca(prot0, evo, data_length)
        prot = torch.cat((prot0, f0agv, evosa), dim=2)
        prot = self.protcnn1(prot.permute(0, 2, 1))
        prot = self.relu(prot)
        prot = self.protcnn2(prot)
        prot = self.relu(prot)
        prot = self.protcnn3(prot)
        prot = self.relu(prot)
        x = self.fc2(prot.permute(0, 2, 1))
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x
#The train() function is designed to train the DeepAIPModule model on bioinformatics datasets
#It utilizes a Focal Loss function for addressing class imbalance and applies Adam optimizer for model parameter updates.
def train():
    train_set = BioinformaticsDataset(prot_train,fusion_train)
    model = DeepAIPModule() #An instance of the DeepAIPModule neural network model, responsible for performing classification on protein sequences.
    epochs = 300            #The number of training iterations
    model = model.to(device) # Ensures that the model is moved to either GPU or CPU for computation.
    train_loader = DataLoader(dataset=train_set, batch_size=256, shuffle=True, num_workers=12, pin_memory=True,
                              persistent_workers=True, collate_fn=coll_paddding)
    best_val_loss = 2000
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Uses the Adam optimizer with a learning rate of 0.0001 to adjust the model's weights during training.
    per_cls_weights = torch.FloatTensor([0.25, 0.75]).to(device) #A tensor that adjusts the class weights, addressing class imbalance by giving more importance to the minority class
    fcloss = FocalLoss_v2(alpha=per_cls_weights, gamma=2) #The Focal Loss function, which focuses on hard-to-classify examples and is useful in imbalanced datasets.
    model.train()
    epochss = []
    losses = []
    for i in range(epochs):
        epoch_loss_train = 0.0
        nb_train = 0
        for prot_x, f0agv, evo_x, data_y, length in train_loader:
            optimizer.zero_grad()
            y_pred = model(prot_x.to(device), f0agv.to(device), evo_x.to(device), length.to(device))
            y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, length.to('cpu'), batch_first=True)
            data_y = torch.nn.utils.rnn.pack_padded_sequence(data_y, length, batch_first=True)
            data_y = data_y.to(device)
            single_loss = fcloss(y_pred.data, data_y.data)
            single_loss.backward()
            optimizer.step()
            epoch_loss_train += single_loss.item()
            nb_train += 1
        epoch_loss_avg = epoch_loss_train / nb_train
        epochss.append(i)
        losses.append(epoch_loss_avg)
        if best_val_loss > epoch_loss_avg:
            model_fn = "DeepAIP.pkl"
            torch.save(model.state_dict(), model_fn)
            best_val_loss = epoch_loss_avg
            if i % 10 == 0:
                print('epochs ', i)
                print("Save model, best_val_loss: ", best_val_loss)

def test():
    #Dataset and DataLoader Initialization
    test_set = BioinformaticsDataset(prot_test,fusion_test)
    test_load = DataLoader(dataset=test_set, batch_size=256, num_workers=12, pin_memory=True, persistent_workers=True, collate_fn=coll_paddding)
    #Model Loading
    model = DeepAIPModule()
    model = model.to(device)
    print("==========================Test RESULT================================")
    model.load_state_dict(torch.load('DeepAIP.pkl'))
    model.eval()
    arr_probs = []
    arr_labels = []
    arr_labels_hyps = []
    with torch.no_grad():
        for prot_x,f0agv,evo_x,data_y, length in test_load:
            y_pred = model(prot_x.to(device),f0agv.to(device),evo_x.to(device),length.to(device))
            y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, length.to('cpu'), batch_first=True)
            y_pred=y_pred.data
            y_pred=torch.nn.functional.softmax(y_pred,dim=1)
            arr_probs.extend(y_pred[:, 1].to('cpu'))
            y_pred=torch.argmax(y_pred, dim=1).to('cpu')
            data_y = torch.nn.utils.rnn.pack_padded_sequence(data_y, length, batch_first=True)
            arr_labels.extend(data_y.data)
            arr_labels_hyps.extend(y_pred)
        auc = metrics.roc_auc_score(arr_labels, arr_probs)
        acc = metrics.accuracy_score(arr_labels, arr_labels_hyps)
        mcc = metrics.matthews_corrcoef(arr_labels, arr_labels_hyps)
        tn, fp, fn, tp = metrics.confusion_matrix(arr_labels, arr_labels_hyps).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1score = 2 * tp / (2 * tp + fp + fn)
        youden = sensitivity + specificity - 1
        #Results Storage and Output
        metrics_dict = {
            'accuracy': metrics.accuracy_score(arr_labels, arr_labels_hyps),
            'balanced_accuracy': metrics.balanced_accuracy_score(arr_labels, arr_labels_hyps),
            'MCC': metrics.matthews_corrcoef(arr_labels, arr_labels_hyps),
            'AUC': metrics.roc_auc_score(arr_labels, arr_probs),
            'AP': metrics.average_precision_score(arr_labels, arr_probs),
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1score,
            'Youden Index': youden
        }
        for key, value in metrics_dict.items():
            print(f'{key}: {value}')

        df = pd.DataFrame([metrics_dict])
        print('<----------------save to csv finish')

    return acc, mcc
if __name__ == "__main__":
    #CUDA Availability Check
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)
    print("use cuda: {}".format(cuda))
    #Dataset Filenames
    device = torch.device("cuda" if cuda else "cpu")
    fusion_train=['prot_t5_pca_gs_equal_train.csv']
    prot_train  =['prot_t5_pca_gs_equal_train.csv']
    fusion_test=['prot-t5_pca_test.csv']
    prot_test  =['prot-t5_pca_test.csv']
    #train()
    test()




