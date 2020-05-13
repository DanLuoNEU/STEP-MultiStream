import os
import sys
import csv
from progress.bar import Bar

def main():
    of_path = "/data/truppr/ava/flows/"
    data_path = "datasets/ava/label/clasp_20200408_validation.csv"
    # print(f"{of_path}")
    # print(f"{data_path}")d
    with open(data_path) as f:
        data_amount = sum(1 for line in f)
    print(data_amount)
    with open(data_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',')
        with Bar('Checking',max=data_amount) as bar:
            for row in spamreader:
                vid_name = row[0]
                timesec = row[1]
                # print(vid_name, timesec.zfill(5))
                for i in range(4):
                    chk_sec = int(timesec) + i - 2 
                    chk_path = of_path + vid_name + '/' + str(chk_sec).zfill(5)
                    if not os.path.exists(chk_path):
                        print('\n' + chk_path + ' | Does NOT exist')
                        continue
                    elif not os.listdir(chk_path):
                        print('\n' + chk_path + ' | Without enough optical files!')

                bar.next()
            # exit(0)

    print("Well Done!")

if __name__ == "__main__":
    main()

