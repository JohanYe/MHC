import pandas as pd

filename = 'model_output.txt'
df = pd.read_csv(filename,sep='\t')
df1 = pd.DataFrame(columns=df.columns)

for allele in df.MHC.unique():
    tmp = df[df.MHC == allele]
    for pep in tmp.Peptide.unique():
        tmp2 = tmp[tmp.Peptide == pep]
        df1.loc[df1.shape[0]] = [tmp2.split.mean(), allele, pep, tmp2.y.unique().item(), tmp2.y_pred.mean()]

df1.to_csv('preprocessed_' + filename, index=False)