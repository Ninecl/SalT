import os
import torch
import logging
import argparse

from dataset import GraphDataset
from evaluator import evaluate


def main(params):
    
    # data_path = os.path.join("./data", params.dataset)
    graph_dataset = GraphDataset(params)
    model = torch.load(params.model_path, map_location=params.device)
    mr, mrr, hit10, hit3, hit1 = evaluate(model, graph_dataset, params.device,
                                          params.query_mode, params.limited_candidate)
    
    
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Test of ImBridge')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="TEST",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB",
                        help="Path to dataset")
    parser.add_argument("--disable_cuda", action='store_true',
                        help="Whether disable the gpu(s)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to use?")
    parser.add_argument("--query_mode", "-m", type=str, default="trans", choices=['trans', 'ind', 'IT'],
                        help="Which mode to test")
    parser.add_argument("--limited_candidate", "-l", type=int, default=None,
                        help="whether use a limited candidate set")
    
    params = parser.parse_args()
    
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')
    
    params.exp_dir = os.path.join("./checkpoint", f"{params.experiment_name}", f"{params.dataset}")
    params.model_path = os.path.join(params.exp_dir, "best.pth")
    assert os.path.exists(params.exp_dir)
    assert os.path.exists(params.model_path)
    # log file
    file_handler = logging.FileHandler(os.path.join(params.exp_dir, f"test_{params.limited_candidate}.log"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    
    main(params)