import os

log_dir = './log_dir'

folders = os.listdir(log_dir)
for f in folders:
    if 'COINRUN' in f:
        path = os.path.join(log_dir, f)
        inters = os.listdir(path)
        print(inters)