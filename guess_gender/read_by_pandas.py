# -*- coding: utf-8 -*-
import pandas as pd


def main():
    data_frame = pd.read_csv("data/gender.txt")
    for index, row in data_frame.iterrows():
        print(row)

if __name__ == "__main__":
    main()
