03/28/2020, Dan

This readme file is for finetuning videos using new annotations and training videos in AVA format.

# 1.Prepare the annotations and videos
## 1.1 Annotations
### Update two files, 
"datasets/ava/label/ava_train_v2.1_ft.csv" is for training \
"datasets/ava/label/ava_val_v2.1_ft.csv" is for simple evaluation

### After updating the annotations, generate pickle files (datasets/ava/label/train.pkl),(datasets/ava/label/val.pkl)
At the root,
```
python scripts/generate_label.py datasets/ava/label/ava_train_v2.1_ft.csv
python scripts/generate_label.py datasets/ava/label/ava_val_v2.1_ft.csv
```
### Overwrite the whole folder
datasets/ava/label -> /data/Dan/ava_v2_1/label

## 1.2 Prepare the video folders
Copy to /data/Dan/ava_v2_1/frames/
- Folder names should be videos' names in annotation files
- Folder names in one video folder should be named after the second indexes of timestamps
- Pictures in one second should .jpg files 

# 2. Finetune the network
## 2.1 Finetune the classification part first
At the root path,
```bash
bash scripts/ft_cls_3.sh
```
All the trained model and logfiles will be under "/data/Dan/ava_v2_1/cache/Cls-max1-i3d-two_branch"

## 2.2 Finetune the whole STEP network
Change the pretrained finetune classification model path in "scripts/ft_step.sh" \

At the root path,  
```bash
bash scripts/ft_step.sh
```
All the trained model and logfiles will be under "/data/Dan/ava_v2_1/cache/Cls-max1-i3d-two_branch"

# 3. Demo
Change "checkpoint_path" to /path/to/pretrained model and "args.data_root" to /path/to/video
- Video folders includes folders named by video name
- Frame files in each video folder should be named from "frame00000.jpg"

```bash
python demo_ft.py
```