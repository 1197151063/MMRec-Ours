# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import yaml
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SMORE', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')

    config_dict = {
        'gpu_id': 0,
    }

    parser.add_argument('--config', help='YAML overrides (grid parameters must be lists)')
    parser.add_argument('--data-path', help='Dataset root containing baby/, sports/, etc.')
    parser.add_argument('--item-id-weight', type=float, help='0 disables the item identity residual')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()
    if args.config:
        with open(args.config, encoding='utf-8') as stream:
            config_dict.update(yaml.safe_load(stream) or {})
    config_dict['gpu_id'] = args.gpu_id
    if args.data_path:
        config_dict['data_path'] = os.path.abspath(args.data_path) + os.sep
    if args.item_id_weight is not None:
        config_dict['item_id_weight'] = [args.item_id_weight]
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


