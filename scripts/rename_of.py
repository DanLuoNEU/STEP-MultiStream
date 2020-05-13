import os
import numpy as np

def main():
    of_path = 'examples/of_frames/cam20-p2p-2-20200427-flows'
    of_list = os.listdir(of_path)
    for of_file in of_list:
        of_name = of_file.split('/')[-1].split('.')[0]
        print(of_name)
        data = np.load(os.path.join(of_path,of_file))
        np.save('examples/of_frames/rename/'+of_name.zfill(4)+'.npy', data)


if __name__ == "__main__":
    main()
