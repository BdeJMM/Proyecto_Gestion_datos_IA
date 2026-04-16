import pandas as pd 
import os 

origen="carpetiña/abandono_escolar_dataset.csv"
destino="data/raw/abandono_escolar_dataset_csv"

os.makedris(destino, exist_ok=True)
df=pd.read_csv("carpetiña/abandono_escolar_dataset.csv")

df.to_csv(os.path.join(destino,"abandono_escolar_dataset.csv"),index=False)