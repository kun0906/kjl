import math
import random
from collections import namedtuple, Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.generate_data import load_flow_data, _generate_datasets


class Environment():

    def step(self, action, y=0, wait_time=0):

        if action == 0 and action == y:  # normal
            reward = 5 + (-wait_time)
            done = True
        elif action == 1 and action == y:  # abnormal
            reward = 10 + (-wait_time)
            done = True
        else:  # action=='wait'  # FP and FN #
            reward = -20 + (-wait_time)
            done = False

        return None, reward, done, None


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NET(nn.Module):

    def __init__(self, in_dim=1, out_dim=1):
        super(NET, self).__init__()

        self.fc1 = nn.Linear(in_dim, 256)  # 6*6 from image dimension
        self.fc2 = nn.Linear(256, out_dim)  # 0 or 1

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


device = 'cpu'
memory = []
GAMMA = 0.9
BATCH_SIZE = 50
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

actions = ['pass', 'alarm', 'none']
n_actions = len(actions)
env = Environment()
steps_done = 0
policy_net = NET(in_dim=40, out_dim=n_actions)
target_net = NET(in_dim=40, out_dim=n_actions)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    v = list([s.view(1, -1) for s in batch.next_state if s is not None])
    if len(v) <= 0:
        return
    non_final_next_states = torch.cat(v, axis=0)
    state_batch = torch.cat([v.view(1, -1) for v in batch.state], axis=0)
    action_batch = torch.cat([v.view(1, -1) for v in batch.action], axis=0)  # the index of actions
    reward_batch = torch.cat([v.view(1, -1) for v in batch.reward], axis=0)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze(1)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # randrange(self, start, stop=None, step=1, _int=int)
        # Choose a random item from range(start, stop[, step]).
        return torch.tensor([[random.randrange(0, stop=n_actions)]], device=device, dtype=torch.long)


def plot_info(info, is_show=True, title=''):
    fig, ax = plt.subplots(nrows=2, ncols=2)

    pkts_info = info['pkts_info']
    rewards_info = info['rewards_info']
    pkts_needed = []
    pkts = []
    for i_eposide, v_lst in enumerate(pkts_info):
        # v = (1, 3) (needed_ptks, all_pkts)
        pkts_needed.append(v_lst[0])
        pkts.append(v_lst[1])

    rewards = []
    for i_eposide, (accumulated, v_lst) in enumerate(rewards_info):
        # rewards: [[-1, -1], [-1, -1, -1], [1], [-1, -1, -1, 1]]
        # rewards.append([sum(v) for v in v_lst])
        rewards.append(accumulated)

    x = range(len(rewards))
    y = np.asarray(pkts_needed).reshape(-1, 1)
    yerr = np.std(y, axis=1)
    y = np.mean(y, axis=1)
    ax[0, 0].errorbar(x, y, yerr, ecolor='r', capsize=2, linestyle='-', marker='.', color='green',
                      markeredgecolor='green', markerfacecolor='green', label=f'pkts_needed',
                      alpha=0.9)  # marker='*',

    # y = np.asarray(pkts).reshape(-1, 1)
    # yerr = np.std(y, axis=1)
    # y = np.mean(y, axis=1)
    # ax[1, 0].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
    #                   markeredgecolor='blue', markerfacecolor='blue', label=f'pkts', alpha=0.9)  # marker='*',
    ax[0, 0].legend(loc='upper right')
    ax[0, 0].set_xlabel('eposide')
    ax[0, 0].set_ylabel('pkts per flow')

    y = np.asarray(rewards).reshape(-1, 1)
    yerr = np.std(y, axis=1)
    y = np.mean(y, axis=1)
    ax[0, 1].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                      markeredgecolor='blue', markerfacecolor='blue', label=f'rewards', alpha=0.9)  # marker='*',
    ax[0, 1].legend(loc='upper right')
    ax[0, 1].set_xlabel('eposide')
    ax[0, 1].set_ylabel('rewards')

    y = np.asarray(pkts).reshape(-1, 1)
    yerr = np.std(y, axis=1)
    y = np.mean(y, axis=1)
    ax[1, 0].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                      markeredgecolor='blue', markerfacecolor='blue', label=f'pkts', alpha=0.9)  # marker='*',
    ax[1, 0].legend(loc='upper right')
    ax[1, 0].set_xlabel('eposide')
    ax[1, 0].set_ylabel('pkts per flow')
    # # plt.xlim([0.0, 1.0])
    # if len(ylim) == 2:
    #     ax.set_ylim(ylim)  # [0.0, 1.05]
    # # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # # plt.xticks(x)
    # # plt.yticks(y)

    # ax.set_title(title)

    fig.suptitle(title, fontsize=11)

    plt.tight_layout()  # rect=[0, 0, 1, 0.95]
    try:
        plt.subplots_adjust(top=0.9, bottom=0.1, right=0.975, left=0.12)
    except Warning as e:
        raise ValueError(e)

    # fig.text(.5, 15, "total label", ha='center')
    # plt.figtext(0.5, 0.01, f'X-axis:({xlabel})', fontsize=11, va="bottom", ha="center")
    # print(out_file)
    # if not pth.exists(os.path.dirname(out_file)): os.makedirs(os.path.dirname(out_file))
    # if pth.exists(out_file): os.remove(out_file)
    # fig.savefig(out_file, format='pdf', dpi=300)
    # out_file += '.png'
    # if pth.exists(out_file): os.remove(out_file)
    # fig.savefig(out_file, format='png', dpi=300)
    if is_show: plt.show()
    plt.close(fig)


def train(X, y):
    TARGET_UPDATE = 5
    info = []
    rewards_info = []
    pkts_info = []
    for i_episode, (X_, y_) in enumerate(zip(X, y)):  # each flow is an episodes

        rewards_ = []
        # each packet is a state
        # state = (torch.Tensor(X_[0]), torch.Tensor([y_]))
        state = torch.Tensor(X_[0]).view(1, -1)
        pkt_0_time = 0
        i_pkt = 0
        accumulated_reward = 0
        done = False
        while True and i_pkt < len(X_):
            action = select_action(state)  # normal: pass, abnormal: alarm
            # perform action and get next state and reward
            # wait_time = (pkt_i.time - pkt_0.time)
            wait_time = i_pkt - pkt_0_time
            _, reward, done, _ = env.step(action.item(), y=y_, wait_time=wait_time)
            accumulated_reward += reward
            rewards_.append((reward, i_pkt))

            if not done:
                if i_pkt + 1 < len(X_):
                    # next_state = (torch.Tensor(X_[i_pkt]), torch.Tensor([y_]))
                    next_state = torch.Tensor(X_[i_pkt + 1]).view(1, -1)
                else:
                    break
            else:
                next_state = None
            # Store the transition in memory
            memory.push(state, torch.tensor([action], dtype=int), next_state, torch.Tensor([reward]))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                # episode_durations.append(i_pkt + 1)
                # plot_durations()
                # plot_rewards(rewards_)
                print(
                    f'i_episode: {i_episode}, accumulated_reward: {accumulated_reward}, i_pkt: {i_pkt}, rewards_: {rewards_}')
                break
            i_pkt += 1
        if not done:
            print(
                f'i_episode: {i_episode}, accumulated_reward: {accumulated_reward}, i_pkt: {i_pkt}, rewards_: {rewards_}, failed!')
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_info.append((accumulated_reward, rewards_))
        pkts_info.append((i_pkt + 1, len(X_) + 1))
    info = {'rewards_info': rewards_info, 'pkts_info': pkts_info}
    print(f'info: {info}')
    plot_info(info, is_show=True, title='train')
    print(f'Complete')


def test(X, y, name='test'):
    TARGET_UPDATE = 5
    info = []
    rewards_info = []
    pkts_info = []
    n_correct = 0
    for i_episode, (X_, y_) in enumerate(zip(X, y)):  # each flow is an episodes

        rewards_ = []
        # each packet is a state
        # state = (torch.Tensor(X_[0]), torch.Tensor([y_]))
        state = torch.Tensor(X_[0]).view(1, -1)
        pkt_0_time = 0
        i_pkt = 0
        accumulated_reward = 0
        done = False
        while True and i_pkt < len(X_):
            action = select_action(state)  # normal: pass, abnormal: alarm
            # perform action and get next state and reward
            # wait_time = (pkt_i.time - pkt_0.time)
            wait_time = i_pkt - pkt_0_time
            _, reward, done, _ = env.step(action.item(), y=y_, wait_time=wait_time)
            accumulated_reward += reward *GAMMA
            rewards_.append((reward, i_pkt))

            if not done:
                if i_pkt + 1 < len(X_):
                    # next_state = (torch.Tensor(X_[i_pkt]), torch.Tensor([y_]))
                    next_state = torch.Tensor(X_[i_pkt + 1]).view(1, -1)
                else:
                    break
            else:
                next_state = None
            # # Store the transition in memory
            # memory.push(state, torch.tensor([action], dtype=int), next_state, torch.Tensor([reward]))

            # Move to the next state
            state = next_state

            # # Perform one step of the optimization (on the target network)
            # optimize_model()
            if done:
                # episode_durations.append(i_pkt + 1)
                # plot_durations()
                # plot_rewards(rewards_)
                print(
                    f'i_episode: {i_episode}, accumulated_reward: {accumulated_reward}, i_pkt: {i_pkt}, rewards_: {rewards_}')
                break
            i_pkt += 1
        if not done:
            print(
                f'i_episode: {i_episode}, accumulated_reward: {accumulated_reward}, i_pkt: {i_pkt}, rewards_: {rewards_}, failed!')
        else:
            n_correct +=1
        # # Update the target network, copying all weights and biases in DQN
        # if i_episode % TARGET_UPDATE == 0:
        #     target_net.load_state_dict(policy_net.state_dict())

        rewards_info.append((accumulated_reward, rewards_))
        pkts_info.append((i_pkt + 1, len(X_) + 1))
    info = {'rewards_info': rewards_info, 'pkts_info': pkts_info}
    print(f'info: {info}')
    plot_info(info, is_show=True, title=name)
    # print(f'Complete')

    acc = n_correct/len(X)
    print(f'{name}_acc: {acc}')
    return acc
#
#
# class RL():
#
#     def __init__(self):
#
#     def train(self, X, y):
#
#
#
#     def test(self, X, y):
#
#


def main():
    is_demo = False
    random_state = 42
    if is_demo:
        X = [
            [[1, 1], [2, 1]],  # one row is one flow, which has multi-packets and each packet has 1500 bytes
            [[2, 2], [2, 2], [4, 4]],

            [[4, 4], [2, 4], [3, 3], [4, 4]],
            [[3, 3]],
        ]
        y = [0, 0, 1, 0]  # normal: 0, abnormal: 1
        X_train, y_train = X, y
        train(X_train, y_train)
    else:
        # data_type = 'payload'  # header, header_payload
        # datasets = _generate_datasets()
        # for v_tup in datasets:
        #     (data_name, data_file), (X, y) = v_tup
        #     print(v_tup)
        X_train, y_train, X_test, y_test = load_flow_data(data_type='header', random_state=random_state)
        # X_train, y_train = X, y
        print(Counter(y_test))
        # in_dim = len(X_train[0][0])

        X_train, y_train = X_test, y_test
        train(X_train, y_train)
        train_acc = test(X_train, y_train, name='evaulate on train set')
        test_acc = test(X_test, y_test, name = 'evaulate on test set')

        print(f'***train_acc:{train_acc}, y_train: {Counter(y_train)}')
        print(f'***test_acc:{test_acc}, y_test: {Counter(y_test)}')

if __name__ == '__main__':
    main()
