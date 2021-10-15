import os
import yaml
import argparse
from datetime import datetime

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import IQNAgent


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent and run.
    agent_temp =  IQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    model_dir = os.path.join(log_dir, 'model/saved')
    for root, directories, files in os.walk(model_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
    if filepath and os.path.exists(filepath):
        agent_temp.load_models(filepath)
    agent = IQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, agent = agent_temp, **config)
    
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'iqn.yaml'))
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
