import pandas as pd
import time

t=time.time()
df=pd.read_csv("race_merged.csv")
print("読み込み秒:",time.time()-t)
print("shape:",df.shape)
