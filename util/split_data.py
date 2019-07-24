"""A generic data loader where the images are arranged in this way: ::

    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/asd932_.png

   pairs文件内容

    500
    Abdullah_Gul	13	14
    Abdullah_Gul	13	16
    Abdullatif_Sener	1	2
    Adel_Al-Jubeir	1	3
    AJ_Lamas	1	Zach_Safrin	1
    Aaron_Guiel	1	Reese_Witherspoon	3
    Aaron_Tippin	1	Jose_Luis_Rodriguez_Zapatero	1
    Abdul_Majeed_Shobokshi	1	Charles_Cope	1

   根据上面的文件，把数据分开的identities和imgs两个文件夹
"""
import shutil
import os
PAIRS_FILE = "/Users/a/Documents/deecamp/datasets/pairs.txt"
ROOT_DIR = "/Users/a/Documents/deecamp/datasets/lfw"
OUTID_DIR = "/Users/a/Documents/deecamp/datasets/lfwid"
OUTIMG_DIR="/Users/a/Documents/deecamp/datasets/lfwtest"
lines = open(PAIRS_FILE,'r').readlines()
number = int(lines[0])
for line in lines[1:number+1]:
    print(line)
    line_split = line.strip().split()
    name = line_split[0]
    dstid_dir = os.path.join(OUTID_DIR,name)
    dstimg_dir = os.path.join(OUTIMG_DIR,name)
    if not os.path.exists(dstid_dir):
        os.mkdir(dstid_dir)
    if not os.path.exists(dstimg_dir):
        os.mkdir(dstimg_dir)
    src_dir = os.path.join(ROOT_DIR,name)
    src_list = os.listdir(src_dir)
    id = int(line_split[1])-1
    img = int(line_split[2])-1
    shutil.copyfile(os.path.join(src_dir,src_list[id]),os.path.join(dstid_dir,src_list[id]))
    shutil.copyfile(os.path.join(src_dir,src_list[img]),os.path.join(dstimg_dir,src_list[img]))
    
