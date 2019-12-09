import os

log_dir = './log_dir'

folders = os.listdir(log_dir)
for f in folders:
    if 'COINRUN' in f:
        inters = os.listdir(f)
        print(inters)