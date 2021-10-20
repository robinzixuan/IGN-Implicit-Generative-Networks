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
    env_online = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)
    print("self.env_online 0:", env_online)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent and run.
    agent_evaluation = IQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
<<<<<<< HEAD
    model_dir = os.path.join(log_dir, 'model/saved')
    filepath = ''
    for root, directories, files in os.walk(model_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
    if os.path.exists(filepath):
        agent_temp.load_models(filepath)
=======


    # load model
    agent_evaluation.load_models(os.path.join("result/PongNoFrameskip-v4/iqn-seed0-20211007-0313/model", "best"))
    print("Model Load done.", args.env_id)

    # 
    print("Start policy evaluation...")

>>>>>>> 89948973a1ee80d1ab842ed694d5e81391f590a7
    agent = IQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, agent=agent_evaluation, env_online=env_online, **config)
    
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'iqn.yaml'))
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)
