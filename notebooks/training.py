import os, sys
import pandas as pd
sys.setrecursionlimit(2147483647)
sys.path.append('/Users/jp/deepcut/deepcut')
sys.path.append('../deepcut')

from train_deepcut import *

generate_best_dataset('input')

model = train_model('cleaned_data')
write_object_to_file(model, 'model.data')
evaluate('cleaned_data', model)
