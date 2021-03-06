import os
import pandas as pd

log_dir = './log_dir'

folders = os.listdir(log_dir)
for f in folders:
    if 'COINRUN' in f:
        path = os.path.join(log_dir, f)
        inters = os.listdir(path)
        if len(inters) > 2:
            try:
                df = [None, None]
                for i in inters:
                    if 'inter-' in i:
                        i_path = os.path.join(path, i, 'progress.csv')
                        df2 = pd.read_csv(i_path)
                        df[1] = df2
                    elif 'inter' in i:
                        i_path = os.path.join(path, i, 'progress.csv')
                        df1 = pd.read_csv(i_path)
                        df[0] = df1
                df = pd.concat(df)
                df.to_csv(os.path.join(path, 'inter', 'progress.csv'), index=False)
            except:
                print(f)
                pass