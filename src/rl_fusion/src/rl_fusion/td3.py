import argparse
import os
import pprint

import gym
import numpy as np
from tianshou import policy
import torch
from std_msgs.msg import String
from tianshou.data import Collector, ReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic
from torch.utils.tensorboard import SummaryWriter

from rl_fusion.env import LvioFusionEnv
from lvio_fusion_node.srv import *

save_net_path = '/home/jyp/Projects/lvio_fusion/misc/td3.pt'
server_init = None
client_create_env = None
client_step = None
stop = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='LvioFusion-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=200)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='/home/jyp/Projects/lvio_fusion/misc/log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--ignore-done', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def train_td3(args=get_args()):
    global client_create_env, client_step
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.layer_num, args.state_shape, device=args.device)
    actor = Actor(net, args.action_shape,args.max_action, args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(args.layer_num, args.state_shape, args.action_shape, concat=True, device=args.device)
    critic1 = Critic(net_c1, args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(args.layer_num, args.state_shape, args.action_shape, concat=True, device=args.device)
    critic2 = Critic(net_c2, args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    policy = TD3Policy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        action_range=[env.action_space.low[0], env.action_space.high[0]],
        tau=args.tau, gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        reward_normalization=args.rew_norm,
        ignore_done=args.ignore_done,
        estimation_step=args.n_step)
    # collector
    train_collector = Collector(policy, train_envs, ReplayBuffer(args.buffer_size))
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, 'td3')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy, save_net_path)

    def stop_fn(mean_rewards):
        global stop
        return stop

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, stop_fn=stop_fn, save_fn=save_fn, writer=writer)
    assert stop_fn(result['best_reward'])
    
    # output
    pprint.pprint(result)
    # Let's watch its performance!
    env = gym.make(args.task)
    policy.eval()
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    print(f'Final reward: {result["rew"]}, length: {result["len"]}')


agent_policy = None

def load_td3(args=get_args()):
    global client_create_env, client_step, agent_policy
    agent_policy = torch.load(save_net_path)
    agent_policy.eval()

def get_weights(obs):
    global agent_policy
    batch = Batch(obs=[obs], info='')  # the first dimension is batch-size
    act = agent_policy(batch).act[0]  # policy.forward return a batch, use ".act" to extract the action
    return act
