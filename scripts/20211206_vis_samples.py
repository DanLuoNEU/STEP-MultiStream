import os
import csv
import cv2
import glob
from PIL import Image
import numpy as np

def read_images(path, videoname, fid, num=36, fps=12):
    """
    Load images from disk for middel frame fid with given num and fps

    return:
        a list of array with shape (num, H,W,C)
    """


    images = []
    list_folders=os.listdir(os.path.join(path, videoname))
    list_folders.sort()
    fid_max=int(list_folders[-1])-1
    
    if num>fps:
        # left of middle frame
        num_left = int(num/2)
        i = 1
        while num_left > 0:
            img_path = os.path.join(path, videoname+'/{:05d}/'.format(max(0,fid-i)))
            images.extend(_load_images(img_path, num=min(num_left, fps), fps=fps, direction='backward'))

            num_left -= fps
            i += 1
        # reverse list
        images = images[::-1]
        
        # img_path = os.path.join(path, videoname+'/{:05d}/'.format(min(fid+i,fid_max)))
        # images.extend(_load_images(img_path, num=fps, fps=fps, direction='forward'))

        # right of middle frame
        num_right = int(np.ceil(num/2))
        i = 0
        while num_right > 0:
            img_path = os.path.join(path, videoname+'/{:05d}/'.format(min(fid+i,fid_max)))
            images.extend(_load_images(img_path, num=min(num_right, fps), fps=fps, direction='forward'))

            num_right -= fps
            i += 1

    return images


def _load_images(path, num, fps=12, direction='forward'):
    """
    Load images in a folder with given num and fps, direction can be either 'forward' or 'backward'
    """

    img_names = glob.glob(os.path.join(path, '*.jpg'))
    if len(img_names) == 0:
        img_names = glob.glob(os.path.join(path, '*.png'))
        if len(img_names) == 0:
            raise ValueError("Image path {} not Found".format(path))
    img_names = sorted(img_names)

    # resampling according to fps
    index = np.linspace(0, len(img_names), fps, endpoint=False, dtype=np.int)
    if direction == 'forward':
        index = index[:num]
    elif direction == 'backward':
        index = index[-num:][::-1]
    else:
        raise ValueError("Not recognized direction", direction)

    images = []
    for idx in index:
        img_name = img_names[idx]
        if os.path.isfile(img_name):
            img = Image.open(img_name)
            images.append(img_name)
        else:
            raise ValueError("Image not found!", img_name)

    return images

def main():
    path_annot = '/data/CLASP-DATA/CLASP2-STEP/data/prepare_data/2_2-annotations/PVD/20211006-3cls-touch_move-xfr_polluted/20211007-PVD-exp1-04162021_08172020-2cls_with_false_alarms-all-Clean.csv'
    dir_frames = '/data/CLASP-DATA/CLASP2-STEP/data/frames'
    dir_outputs = '/home/dan/ws/STEP-MultiStream/test/vis/annotation'
    if not os.path.exists(dir_outputs):
        os.makedirs(dir_outputs)

    fps=10
    ids_act=[1, 2, 3]
    id_annot = [1,1,1]
    with open(path_annot, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if int(row[6]) in ids_act:
                name_vid=row[0]
                sec_vid =int(row[1])
                bbox=[row[2],row[3],row[4],row[5]]
                paths_images=read_images(dir_frames, name_vid, sec_vid, num=36, fps=10)

                img=[]
                for path_img in paths_images:
                    im=Image.open(path_img)
                    width,height=im.size
                    im_crop=im.crop((float(row[2])*width,float(row[3])*height,float(row[4])*width,float(row[5])*height))
                    img.append(im_crop)
                    
                img[0].save(fp=os.path.join(dir_outputs,f'{row[6]}_{id_annot[int(row[6])-1]}_{sec_vid}_{row[7]}.gif'),format='GIF',append_images=img,save_all=True,duration=100,loop=0)
                # ids_act.remove(int(row[6]))
                print(os.path.join(dir_outputs,f'{row[6]}_{id_annot[int(row[6])-1]}_{sec_vid}_{row[7]}.gif'))
                id_annot[int(row[6])-1] += 1
                
            pass


    pass

if __name__ == '__main__':
    main()