import argparse
import os

### Configurations
parser = argparse.ArgumentParser()

### General Parameters 
parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--project-name', default='test')

### Training Parameters

### Dataset Parameters
parser.add_argument('--train-data-path', type=str, default="/home/seyun/text_categorization/TC_provided/corpus1_train.labels")
parser.add_argument('--test-data-path', type=str, default="/home/seyun/text_categorization/TC_provided/corpus1_test.labels")

args = parser.parse_args()