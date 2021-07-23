#!/usr/bin/env python3

'''
This script exctracts training variables from all logs from 
tensorflow event files ("event*"), writes them to Pandas 
and finally stores in long-format to a CSV-file including
all (readable) runs of the logging directory.
'''

import tensorflow as tf
import glob
import os
import pandas as pd


# Get all event* runs from logging_dir subdirectories
logging_dir = '/home/gyan/BoschServer/data2/datasets/mml/soundspaces/models/savn_sp/tb'
event_paths = glob.glob(os.path.join(logging_dir, "*event*"))


# Extraction function
def sum_log(path):
    runlog = pd.DataFrame(columns=['metric', 'value'])
    metric_set = set()
    try:
        for e in tf.train.summary_iterator(path):
            # print("e.summary: ", e.summary)
            # exit()
            for v in e.summary.value:
                # print("v: ", v)
                r = {'metric': v.tag, 'value':v.simple_value}
                runlog = runlog.append(r, ignore_index=True)

                if v.tag not in metric_set:
                    metric_set.add(v.tag)
    
    # Dirty catch of DataLossError
    except:
        print('Event file possibly corrupt: {}'.format(path))
        return None

    num_metric = len(metric_set)
    runlog['epoch'] = [item for sublist in [[i]*num_metric for i in range(0, len(runlog)//num_metric)] for item in sublist]
    
    return runlog


# Call & append
all_log = pd.DataFrame()
for path in event_paths:
    log = sum_log(path)
    if log is not None:
        if all_log.shape[0] == 0:
            all_log = log
        else:
            all_log = all_log.append(log)


# Inspect
print(all_log.shape)
all_log.head()    
            
# Store
all_log.to_csv('all_training_logs_in_one_file.csv', index=None)