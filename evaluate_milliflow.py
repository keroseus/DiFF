import argparse

import yaml

from evaluate import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a MilliFlow checkpoint with the training-time 6/4 KNN setting.'
    )
    parser.add_argument('--config', default='config_milliflow.yaml')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional smoke-test limit; point clouds are still evaluated at raw size.',
    )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    if config.get('dataset_name', '').lower() != 'milliflow':
        raise ValueError('evaluate_milliflow.py requires dataset_name: milliflow')

    feature_k = int(config.get('feature_kneighbors', 6))
    correlation_k = int(config.get('correlation_kneighbors', 4))
    if (feature_k, correlation_k) != (6, 4):
        raise ValueError(
            'MilliFlow checkpoint evaluation requires feature/correlation KNN = 6/4, '
            f'but the config specifies {feature_k}/{correlation_k}.'
        )

    config['feature_kneighbors'] = feature_k
    config['correlation_kneighbors'] = correlation_k
    config['test_partition'] = 'test'
    evaluate(config, args.ckpt, args.max_samples)
