# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger

from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import json
from pathlib import Path
import os
import torch



          

def quick_start(model, dataset, config_dict, save_model=False, mg=False):
    # merge config dict
    config = Config(model, dataset, config_dict, mg)

    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    train_df = train_dataset.df
    train_edge_index = torch.tensor(train_df[['userID', 'itemID']].values.T, dtype=torch.long)

    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    run_records = []
    best_valid_value = float('-inf') if config['valid_metric_bigger'] else float('inf')
    idx = best_config_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        value = config[i]
        hyper_ls.append(value if isinstance(value, list) else [value])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model, mg)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, 
                                                                                valid_data=valid_data, 
                                                                                test_data=test_data, 
                                                                                saved=save_model,
                                                                                train_edge_index=train_edge_index,
                                                                                num_users=dataset.user_num)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))
        run_records.append({
            'model': config['model'], 'dataset': config['dataset'],
            'parameters': dict(zip(config['hyper_parameters'], hyper_tuple)),
            'best_epoch': getattr(trainer, 'best_epoch', None),
            'valid': best_valid_result, 'test': best_test_upon_valid})
        if config['result_file']:
            result_path = Path(config['result_file'])
            result_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = result_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(run_records, indent=2), encoding='utf-8')
            temporary.replace(result_path)

        # Select hyperparameters using validation only.
        improved = (best_valid_score > best_valid_value if config['valid_metric_bigger']
                    else best_valid_score < best_valid_value)
        if improved:
            best_valid_value = best_valid_score
            best_config_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_config_idx][0], dict2str(hyper_ret[best_config_idx][1]), dict2str(hyper_ret[best_config_idx][2])))
        

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_config_idx][0],
                                                                   dict2str(hyper_ret[best_config_idx][1]),
                                                                   dict2str(hyper_ret[best_config_idx][2])))

