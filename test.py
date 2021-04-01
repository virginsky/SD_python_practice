import numpy as np
import pandas as pd
import matplotlib as plt

# print(np.__version__)
# print(pd.__version__)
# print(plt.__version__)



# df1 = pd.DataFrame(np.arange(16).reshape(4,-1), 
#                     columns = ('c1','c2','c3','c4'), 
#                     index = ('r1','r2','r3','r4'))

# print(df1)
# df1.drop('c2',axis = 1) #칼럼으로 접근
# print(df1.drop(columns = ['c3','c4']))

# merge 데이터병합

# data1 = pd.DataFrame({'id':['01','02','03','04','05','06'],
#                     'col1': np.random.randint(0,50,6),  
#                     'col2': np.random.randint(1000,2000,6)})

# data2 = pd.DataFrame({'id':['04','05','06','07'],
#                     'col1': np.random.randint(0,50,4),  
#                     'col2': np.random.randint(1000,2000,4)})

# print(data1, data2)
# print(pd.merge(data1,data2, on='id'))
# print(pd.merge(data1,data2, how = 'inner', on='id')) #교집합
# print(pd.merge(data1,data2, how = 'outer', on='id')) #합집합
# print(pd.merge(data1,data2, how = 'right', on='id'))
df = pd.read_csv('C:/Users/sundooedu/Documents/JH_version/Day16/kbo.csv')
# df
# print(df.max()) # default -> axis = 1(행) -> axis = 0(열)
# print(df.min()) # default -> axis = 1(행) -> axis = 0(열)

print(df.isnull().sum().sum())
print(df.isna().sum().sum())