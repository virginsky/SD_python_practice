import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## p28 (Code) 수학 코딩해보기
x = [a for a in range(6)]
f = [[-x*x + 4*x + 1 for b in range(6)]]
plt.plot(x, f)      #구간 : x = 0~5까지
plt.show()