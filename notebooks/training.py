import os, sys
import pandas as pd
sys.setrecursionlimit(2147483647)
sys.path.append('/Users/jp/deepcut/deepcut')
sys.path.append('../deepcut')

from train_deepcut import *
log("GENERATE DATA SET")
generate_best_dataset('input')

log("TRAIN MODEL")
model = train_model('cleaned_data')
log("TRAIN MODEL COMPLETE")
write_object_to_file(model, 'model.data')
log("EVALUATE")
f1score, precision, recall = evaluate('cleaned_data', model)
log("F1 {0}".format(f1score))
log("precision {0}".format(precision))
log("recall {0}".format(recall))
