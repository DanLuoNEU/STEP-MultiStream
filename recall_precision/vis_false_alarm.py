import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

font = ImageFont.truetype("utils/arial.ttf", 28)

def draw_rectangle(draw, coords, outline, width=4):
    for i in range(int(width/2)):
        draw.rectangle(coords-i, outline=outline, fill=None)
    for i in range(1,int(width/2)):
        draw.rectangle(coords+i, outline=outline, fill=None)

def main():
    path_false_alarm='/home/dan/ws/STEP-MultiStream/test/kriMixed-1cls-test/Recall_Precision/numbers_conf07.txt'
    path_vid='/data/CLASP-DATA/CLASP2-STEP/data/frames'
    path_save='/home/dan/ws/STEP-MultiStream/test/kriMixed-1cls-test/Recall_Precision/vis_conf07'

    with open(path_false_alarm,'r') as f:
        lines=f.readlines()

        for line in lines:
            if line.startswith('20191024'):
                print(line)
                name_video,sid,x0,y0,x1,y1,l,s = line.split(',')
                path_sec=os.path.join(path_vid, name_video,'0'+sid)
                path_output_sec=os.path.join(path_save,name_video,sid)
                if not os.path.exists(path_output_sec):
                    os.makedirs(path_output_sec)
                for name_image in os.listdir(path_sec):
                    path_img=os.path.join(path_sec, name_image)
                    I = Image.open(path_img)
                    W, H = I.size

                    draw = ImageDraw.Draw(I)
                    
                    box = np.asarray([x0,y0,x1,y1], dtype=np.float32)
                    box[::2] *= W
                    box[1::2] *= H
                    draw_rectangle(draw, box, outline="red") # One bounding box for one result

                    path_output = os.path.join(path_output_sec,name_image)
                    I.save(path_output)

    pass

if __name__ == '__main__':
    main()