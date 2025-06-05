import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('./data/student_habits_performance.csv')
    X = df.iloc[:, 1: -1]
    y = df.iloc[:, -1]
    

if __name__ == '__main__':
    main()



