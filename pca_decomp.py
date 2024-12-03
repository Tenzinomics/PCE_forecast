from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


def pca_model(df_clean):
    # Standardize the data (mean=0, variance=1)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)

    #Apply PCA 
    pca = PCA(n_components=51) #The component has to be the number of series used.
    principal_components = pca.fit_transform(df_scaled)

    #Getting the eigen values 
    eigenvalues = pca.explained_variance_

    #creating a dataframe with the index
    pca_df = pd.DataFrame(data=principal_components)
    pca_df.index = df_clean.index.values
    pca_df.columns = [f'PC{i+1}' for i in range(len(df_clean.columns))]

    return pca_df, eigenvalues

