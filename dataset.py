import torch

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, df,SEQUENCE_LEN):
        'Initialization'
        self.seq_len = SEQUENCE_LEN
        self.df = df

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.df) - self.seq_len - 1

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        df1 = self.df.iloc[index:index+self.seq_len]

        MEAN = df1.mean()
        STD = df1.std()
        # print(df1.shape,MEAN.shape)
      
        # Normalize the training features
        df1 = (df1 - MEAN) / STD
        

        y = self.df.iloc[index+self.seq_len+1,0]
        y = (y-MEAN[0])/ STD[0]
        
        return torch.tensor(df1.values,dtype=torch.float32), torch.tensor(y, dtype=torch.float32), MEAN[0],STD[0]