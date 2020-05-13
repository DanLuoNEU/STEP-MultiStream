import os
import sys
import csv
import random


infile = "datasets/ava/label/ava_val_v2.1.csv"
outfile = "datasets/ava/label/ava_val_v2.1_ft.csv"


def main():
    row_give, row_take, row_bg = [], [], []
    with open(infile) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if row[6] == '65':
                row[6] = '1'
                row_give.append(row)
            elif row[6] == '78':
                row[6] = '2'
                row_take.append(row)
            else:
                row[6] = '３' # 0 would cause some evaluation problem
                row_bg.append(row)
    if len(row_give) < len(row_take):
        num_sam = len(row_give) 
        row_take = random.choices(row_take,k=num_sam)
    else:
        num_sam = len(row_take) 
        row_give = random.choices(row_give,k=num_sam)
    row_bg_rdm = random.choices(row_bg,k=num_sam)
    print(len(row_take))
    row_ft_all = row_bg_rdm + row_give + row_take
    print(len(row_ft_all))
    with open(outfile,'w', newline='') as file:
        writer = csv.writer(file)
        for row in row_ft_all:
            writer.writerow(row)
    
    
if __name__ == "__main__":
    main()