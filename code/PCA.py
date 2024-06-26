import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('../Dataset/enhance/prot-t5_gs_train.csv')
test_df  = pd.read_csv('../Dataset/prot-t5_test.csv')
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
features_pca = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
train_pca_df = pd.DataFrame(X_train_pca, columns=features_pca)
train_pca_df.insert(0, 'label', y_train)
test_pca_df = pd.DataFrame(X_test_pca, columns=features_pca)
test_pca_df.insert(0, 'label', test_df['label'])
train_pca_df.to_csv('../DataSet/enhance/prot-t5_gs_pca_train.csv', index=False)
test_pca_df.to_csv( '../DataSet/enhance/prot-t5_gs_pca_test.csv', index=False)
