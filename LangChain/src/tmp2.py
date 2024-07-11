import pandas as pd
import numpy as np

input_df = pd.read_csv("/home/Chatbot/LangChain/assets/open/test.csv")

input_df = input_df.head(30)

print(input_df['question'])