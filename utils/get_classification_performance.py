import argparse

# from eval_classification import ANETclassification
from eval_ucf101 import UCFclassification

def main(ground_truth_filename, prediction_filename,
         subset='validation', verbose=True, top_k=1):
    ucf_net_eval = UCFclassification(ground_truth_filename, 
                                    prediction_filename, 
                                    subset, 
                                    verbose, 
                                    # top_k)
                                    1)
    ucf_net_eval.evaluate()
    ucf_net_eval = UCFclassification(ground_truth_filename, 
                                    prediction_filename, 
                                    subset, 
                                    verbose, 
                                    # top_k)
                                    5)
    ucf_net_eval.evaluate()

def parse_input():
    description = ('This script allows you to evaluate the UCF101 '
                   'untrimmed video classification task which is intended to '
                   'evaluate the ability of algorithms to predict activities '
                   'in untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('ground_truth_filename',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('prediction_filename',
                   help='Full path to json file containing the predictions.')
    p.add_argument('--subset', default='validation',
                   help=('String indicating subset to evaluate: '
                         '(training, validation)'))
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--top_k', type=int, default=1)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_input()
    main(**vars(args))
