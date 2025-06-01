import argparse
import math
import time

import numpy as np
import torch
from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from RL.Env import ClassifyEnv
from RL.agent import DQNAgent
from RL.agent_priority import DQNAgent_PER

# 创建参数解析器
parser = argparse.ArgumentParser(description='Save arguments to a file')
parser.add_argument('--imb_rate', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--epsilon_max', type=float, default=1)
parser.add_argument('--epsilon_min', type=float, default=0.1)
parser.add_argument('--eps_decay', type=int, default=40000)
parser.add_argument('--steps', type=int, default=40000)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--max_buff', type=int, default=100000)
parser.add_argument('--update_tar_interval', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--print_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=1000)
parser.add_argument('--learning_start', type=int, default=10000)
args = parser.parse_args()
imb_rate = args.imb_rate
USE_CUDA = torch.cuda.is_available()


def plot_training(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.title('steps %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


def get_current_value(step):
    a = -float(args.epsilon_max - args.epsilon_min) / float(args.eps_decay)
    b = float(args.epsilon_max)
    value = max(args.epsilon_min, a * float(step) + b)
    return value


def one_train(fold, X_train, y_train, X_test, y_test):
    mode = 'train'
    env = ClassifyEnv(mode, imb_rate, X_train, y_train)

    action_space = env.action_space
    agent = DQNAgent(action_space, USE_CUDA, lr=args.learning_rate, memory_size=args.max_buff)

    state = env.reset()

    episode_reward = 0
    all_rewards = []
    losses = []
    episode_num = 0
    done = False
    # tensorboard
    starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
    print("Start experiment:", starttime)
    summary_writer = SummaryWriter(log_dir="./logs/" + starttime[:16].replace(":", "_"),
                                   comment=starttime[:16].replace(":", "_"), flush_secs=60)
    writer = SummaryWriter(log_dir="./logs/" + starttime[:16].replace(":", "_"))

    # 把超参数设置添加进去
    for arg in vars(args):
        writer.add_text(arg, str(getattr(args, arg)), global_step=0)

    # 关闭 SummaryWriter
    writer.close()

    # e-greedy decay
    epsilon_by_frame = lambda frame_idx: args.epsilon_min + (args.epsilon_max - args.epsilon_min) * math.exp(
        -1. * frame_idx / args.eps_decay)

    for i in range(args.steps):
        epsilon = epsilon_by_frame(i)
        # epsilon = get_current_value(i)
        state_tensor = agent.observe(state)
        action, q_value = agent.act(state_tensor, epsilon)

        next_state, reward, done, _ = env.step(action, q_value)

        episode_reward += reward
        agent.memory_buffer.push(state, action, reward, next_state, done)
        state = next_state

        loss = 0
        if agent.memory_buffer.size() >= args.learning_start and i % 1 == 0:
            loss = agent.learn_from_experience(args.batch_size, args.gamma)
            losses.append(loss)

        if i % args.print_interval == 0:
            print("steps: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (
                i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
            summary_writer.add_scalar("Temporal Difference Loss", loss, i)
            summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
            summary_writer.add_scalar("Epsilon", epsilon, i)

        if i % args.update_tar_interval == 0:
            agent.DQN_target.load_state_dict(agent.DQN.state_dict())

        if done:
            summary_writer.add_scalar("Episode Reward", episode_reward, episode_num)
            state = env.reset()

            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_num += 1
            avg_reward = float(np.mean(all_rewards[-100:]))

    summary_writer.close()
    torch.save(agent.DQN_target.state_dict(), "../save_cmdc/imb0.5_{}.pth".format(fold))
    # plot_training(args.steps, all_rewards, losses)

    mode = 'test'
    env = ClassifyEnv(mode, imb_rate, X_train, y_train)
    # action_space = env.action_space
    # agent = DQNAgent(action_space, USE_CUDA, lr=args.learning_rate, memory_size=args.max_buff)
    done = False
    train_state = env.reset()
    while done is not True:
        state_tensor = agent.observe(train_state)
        action, q_value = agent.act(state_tensor, epsilon=0)
        next_train_state, reward, done, _ = env.step(action, q_value)
        train_state = next_train_state

    print('*' * 20)
    print('开始测试')
    print('*' * 20)

    mode = 'test'
    env = ClassifyEnv(mode, imb_rate, X_test, y_test)
    # action_space = env.action_space
    # agent = DQNAgent(action_space, USE_CUDA, lr=args.learning_rate, memory_size=args.max_buff)
    done = False
    test_state = env.reset()
    while done is not True:
        state_tensor = agent.observe(test_state)
        action, q_value = agent.act(state_tensor, epsilon=0)
        next_test_state, reward, done, _ = env.step(action, q_value)
        test_state = next_test_state

    info = env.get_info()
    print('*' * 20, '片段级结果', '*' * 20)
    print('rl acc:', info['acc'])
    print('rl precision:', info['precision'])
    print('rl recall:', info['recall'])
    print('rl f1:', info['fmeasure'])
    print('rl auc:', info['auc'])
    return info['acc'], info['precision'], info['recall'], info['fmeasure'], info['auc'], info['cm']
