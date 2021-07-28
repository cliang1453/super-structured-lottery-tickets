import os
import argparse
import random
from sys import path

path.append(os.getcwd())
from experiments.common_utils import dump_rows
from data_utils.task_def import DataFormat
from data_utils.log_wrapper import create_logger
from experiments.glue.glue_utils import *

logger = create_logger(__name__, to_disk=True, log_file='split_prepro.log')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--old_glue', action='store_true', help='whether it is old GLUE, refer official GLUE webpage for details')
    parser.add_argument('--split_task', type=str, default='sst')
    parser.add_argument('--split_rate', type=int, default=1)
    args = parser.parse_args()
    return args

def main(args):
    is_old_glue = args.old_glue
    root = args.root_dir
    assert os.path.exists(root)
    canonical_data_suffix = "canonical_data_split" + str(args.split_rate)
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    if args.split_task == "sst":
        sst_train_path = os.path.join(root, 'SST-2/train.tsv')
        sst_dev_path = os.path.join(root, 'SST-2/dev.tsv')
        sst_test_path = os.path.join(root, 'SST-2/test.tsv')

        sst_train_data = load_sst(sst_train_path, split=args.split_rate)
        logger.info('Loaded {} SST train samples'.format(len(sst_train_data)))

        sst_train_fout = os.path.join(canonical_data_root, 'sst_train.tsv')
        dump_rows(sst_train_data, sst_train_fout, DataFormat.PremiseOnly)
        logger.info('done with sst')
    elif args.split_task == "mnli":

        multi_train_path = os.path.join(root, 'MNLI/train.tsv')
        multi_dev_matched_path = os.path.join(root, 'MNLI/dev_matched.tsv')
        multi_dev_mismatched_path = os.path.join(root, 'MNLI/dev_mismatched.tsv')

        multinli_train_data = load_mnli(multi_train_path, split=args.split_rate)
        multinli_matched_dev_data = load_mnli(multi_dev_matched_path)
        multinli_mismatched_dev_data = load_mnli(multi_dev_mismatched_path)

        logger.info('Loaded {} MNLI train samples'.format(len(multinli_train_data)))
        logger.info('Loaded {} MNLI matched dev samples'.format(len(multinli_matched_dev_data)))
        logger.info('Loaded {} MNLI mismatched dev samples'.format(len(multinli_mismatched_dev_data)))

        multinli_train_fout = os.path.join(canonical_data_root, 'mnli_train.tsv')
        multinli_matched_dev_fout = os.path.join(canonical_data_root, 'mnli_matched_dev.tsv')
        multinli_mismatched_dev_fout = os.path.join(canonical_data_root, 'mnli_mismatched_dev.tsv')
        dump_rows(multinli_train_data, multinli_train_fout, DataFormat.PremiseAndOneHypothesis)
        dump_rows(multinli_matched_dev_data, multinli_matched_dev_fout, DataFormat.PremiseAndOneHypothesis)
        dump_rows(multinli_mismatched_dev_data, multinli_mismatched_dev_fout, DataFormat.PremiseAndOneHypothesis)
        logger.info('done with mnli')




if __name__ == '__main__':
    args = parse_args()
    main(args)
