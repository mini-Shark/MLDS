import pandas as pd 
import glob 
import os 
import os.path as osp 
# 以后在windows里写路径都用这形式，不要再加‘r’了
image_root = "D:\\test_data\\cifar-10\\train"
csv_root = "D:\\test_data\\cifar-10\\train\\data_pair.csv"
# 这种带if的列表推导式有两种形式， 单个if的情况格式如下
# if-else 同时存在时放在for前面
subfolders = [i for i in os.listdir(image_root) if osp.isdir(osp.join(image_root, i ))] 
all_data_pair = [] 

for folder in subfolders:
  all_files = glob.glob(osp.join(image_root,folder,"*.jpg"))
  label = int(folder)
  tmp_list = [[file, label]for file in all_files]
  all_data_pair.extend(tmp_list)

df = pd.DataFrame(all_data_pair, columns=['path', 'label'])
df.to_csv(csv_root, index=False)
