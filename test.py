import numpy as np
import pandas as pd
import matplotlib as plt

print(np.__version__)
print(pd.__version__)
print(plt.__version__)

df = pd.read_csv('C:/Users/sundooedu/Documents/JH_version/Day16/kbo.csv')
df
print(df.max()) # default -> axis = 1(행) -> axis = 0(열)
print(df.min()) # default -> axis = 1(행) -> axis = 0(열)

df1 = pd.DataFrame(np.arange(16).reshape(4,-1), 
                    columns = ('c1','c2','c3','c4'), 
                    index = ('r1','r2','r3','r4'))

print(df1)
df1.drop('c2',axis = 1) #칼럼으로 접근
df1