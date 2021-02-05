import math
import random
from collections import namedtuple, Counter

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from rl.generate_data import load_flow_data, balance_data

device = 'cpu'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def plot_info(info, is_show=True, title=''):
    fig, ax = plt.subplots(nrows=3, ncols=3)

    pkts_info = info['pkts_info']
    rewards_info = info['rewards_info']
    y_info = info['y_info']
    pkts_needed = []
    pkts = []
    for i_episode, v_lst in enumerate(pkts_info):
        # v = (1, 3) (needed_ptks, all_pkts)
        pkts_needed.append(v_lst[0])
        pkts.append(v_lst[1])

    rewards = []
    for i_episode, (accumulated, v_lst) in enumerate(rewards_info):
        # rewards: [[-1, -1], [-1, -1, -1], [1], [-1, -1, -1, 1]]
        # rewards.append([sum(v) for v in v_lst])
        rewards.append(accumulated)

    fontsize = 7
    x = range(len(rewards))
    y = np.asarray(pkts_needed).reshape(-1, 1)
    yerr = np.std(y, axis=1)
    y = np.mean(y, axis=1)
    y_norm = [v for (v, (y_, y_pred_)) in zip(y, y_info) if y_ == 0]
    y_abnorm = [v for (v, (y_, y_pred_)) in zip(y, y_info) if y_ == 1]
    print(f'average_needed_pkts_per_flow: {np.mean(y)}+/-{np.std(y)}, in which, norm: '
          f'{np.mean(y_norm)}+/-{np.std(y_norm)}, abnorm: {np.mean(y_abnorm)}+/-{np.std(y_abnorm)}')
    ax[0, 0].errorbar(x, y, yerr, ecolor='r', capsize=2, linestyle='-', marker='.', color='green',
                      markeredgecolor='green', markerfacecolor='green', label=f'pkts_needed',
                      alpha=0.9)  # marker='*',

    # y = np.asarray(pkts).reshape(-1, 1)
    # yerr = np.std(y, axis=1)
    # y = np.mean(y, axis=1)
    # ax[1, 0].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
    #                   markeredgecolor='blue', markerfacecolor='blue', label=f'pkts', alpha=0.9)  # marker='*',
    ax[0, 0].legend(loc='upper right', fontsize=fontsize)
    ax[0, 0].set_xlabel('episode')
    ax[0, 0].set_ylabel('pkts per flow')

    y = np.asarray(pkts).reshape(-1, 1)
    yerr = np.std(y, axis=1)
    y = np.mean(y, axis=1)
    y_norm = [v for (v, (y_, y_pred_)) in zip(y, y_info) if y_ == 0]
    y_abnorm = [v for (v, (y_, y_pred_)) in zip(y, y_info) if y_ == 1]
    print(f'average_pkts_per_flow: {np.mean(y)}+/-{np.std(y)}, in which, norm: '
          f'{np.mean(y_norm)}+/-{np.std(y_norm)}, abnorm: {np.mean(y_abnorm)}+/-{np.std(y_abnorm)}')
    ax[0, 1].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                      markeredgecolor='blue', markerfacecolor='blue', label=f'pkts', alpha=0.9)  # marker='*',
    ax[0, 1].legend(loc='upper right', fontsize=fontsize)
    ax[0, 1].set_xlabel('episode')
    ax[0, 1].set_ylabel('pkts per flow')

    y = np.asarray(rewards).reshape(-1, 1)
    yerr = np.std(y, axis=1)
    y = np.mean(y, axis=1)
    y_norm = [v for (v, (y_, y_pred_)) in zip(y, y_info) if y_ == 0]
    y_abnorm = [v for (v, (y_, y_pred_)) in zip(y, y_info) if y_ == 1]
    print(f'average_reward: {np.mean(y)}+/-{np.std(y)}, in which, norm: '
          f'{np.mean(y_norm)}+/-{np.std(y_norm)}, abnorm: {np.mean(y_abnorm)}+/-{np.std(y_abnorm)}')
    ax[1, 0].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                      markeredgecolor='blue', markerfacecolor='blue', label=f'rewards', alpha=0.9)  # marker='*',
    ax[1, 0].legend(loc='upper right', fontsize=fontsize)
    ax[1, 0].set_xlabel('episode')
    ax[1, 0].set_ylabel('rewards')
    ax[1, 0].set_ylim([-50, 11])  # [0.0, 1.05]

    step = 100
    if 'train_phase' in title:
        train_loss_info = info['train_loss_info']
        y = np.asarray(train_loss_info).reshape(-1, 1)
        yerr = np.std(y, axis=1)
        y = np.mean(y, axis=1)
        ax[1, 1].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                          markeredgecolor='blue', markerfacecolor='blue', label=f'training_loss',
                          alpha=0.9)  # marker='*',
        ax[1, 1].legend(loc='upper right', fontsize=fontsize)
        ax[1, 1].set_xlabel('episode')
        ax[1, 1].set_ylabel('training_loss')

        # average loss for different episodes
        y = [np.mean(train_loss_info[0:i + step]) for i in range(0, len(train_loss_info), step)]
        x = range(len(y))
        y = np.asarray(y).reshape(-1, 1)
        yerr = np.std(y, axis=1)
        y = np.mean(y, axis=1)
        ax[1, 2].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                          markeredgecolor='blue', markerfacecolor='blue', label=f'average_loss',
                          alpha=0.9)  # marker='*',
        ax[1, 2].legend(loc='upper right', fontsize=fontsize)
        ax[1, 2].set_xlabel(f'average (step={step})')
        ax[1, 2].set_ylabel('average training loss')
        # ax[2, 0].set_ylim([-50, 11])  # [0.0, 1.05]

    # average rewards for different episodes
    rewards = [np.mean(rewards[0:i+step]) for i in range(0, len(rewards), step)]
    x = range(len(rewards))
    y = np.asarray(rewards).reshape(-1, 1)
    yerr = np.std(y, axis=1)
    y = np.mean(y, axis=1)
    ax[2, 0].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                      markeredgecolor='blue', markerfacecolor='blue', label=f'average_rewards', alpha=0.9)  # marker='*',
    ax[2, 0].legend(loc='upper right', fontsize=fontsize)
    ax[2, 0].set_xlabel(f'average (step={step})')
    ax[2, 0].set_ylabel('average rewards')
    # ax[2, 0].set_ylim([-50, 11])  # [0.0, 1.05]

    # average pkts_needed for different episodes
    pkts_needed = [np.mean(pkts_needed[0:i + step]) for i in range(0, len(pkts_needed), step)]
    x = range(len(pkts_needed))
    y = np.asarray(pkts_needed).reshape(-1, 1)
    yerr = np.std(y, axis=1)
    y = np.mean(y, axis=1)
    ax[2, 1].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                      markeredgecolor='blue', markerfacecolor='blue', label=f'average_pkts_needed',
                      alpha=0.9)  # marker='*',
    ax[2, 1].legend(loc='upper right', fontsize=fontsize)
    ax[2, 1].set_xlabel(f'average (step={step})')
    ax[2, 1].set_ylabel('average pkts_needed')
    # ax[2, 0].set_ylim([-50, 11])  # [0.0, 1.05]

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


def reduce_reward(reward, init=10, ith_wait_time=0, is_recongized=True):
    print(f'func', ith_wait_time, reward,ith_wait_time == 0 , flush=True)

    if ith_wait_time == 0:
        reward = init
    else:
        if is_recongized:
            reward -= reward / 2
        else:
            reward -= reward * 2
    # print(reward)
    return reward


class Environment():

    def reset(self):
        pass

    def step(self, action, y=0, wait_time=0, reward_method='divide'):

        # action = [0, 1, 2]

        ### reward_method 1
        if reward_method == 'add':
            if action == y:
                if action == 0:  # normal
                    reward = 10 + (-wait_time)  # reward -= reward/2
                    done = True
                else:  # action == 1:  # abnormal
                    reward = 5 + (-wait_time)
                    done = True
            else:  # action=='wait'  # FP and FN #
                if action == 0 and y == 1:  # FN
                    reward = -2*((wait_time+1))
                else:  # action == 1 and y == 0: FP, 0 is normal and 1 is abnormal
                    reward = -1*((wait_time+1))
                done = False
        elif reward_method == 'divide':
            # print(wait_time, reward, action, y, flush=True)
            if action == y:
                if action == 0:  # normal
                    reward = 10 * (1/2**wait_time)
                    done = True
                else:  # action == 1:  # abnormal
                    reward = 5 * (1/2**wait_time)
                    done = True
            else:  # action=='wait'  # FP and FN #action = [0, 1, 2]
                if action == 0 and y == 1:  # FN
                    # reward = -2 * (2**wait_time)
                    reward = -2 * ((wait_time+1))
                else:  # action == 1 and y == 0: FP, 0 is normal and 1 is abnormal
                    reward = -1 * ((wait_time+1))
                done = False
        elif reward_method == 'demo':   # try to use as many as pkts
            if action == y:
                if action == 0:  # normal
                    reward = 10 * (2**wait_time)
                    done = True
                else:  # action == 1:  # abnormal
                    reward = 5 * (2**wait_time)
                    done = True
            else:  # action=='wait'  # FP and FN #action = [0, 1, 2]
                if action == 0 and y == 1:  # FN
                    reward = -1 * (1/2**wait_time)
                else:  # action == 1 and y == 0: FP, 0 is normal and 1 is abnormal
                    reward = -1 * (1/2**wait_time)
                done = False

        else:
            msg = f'{reward_method}'
            raise NotImplementedError(msg)

        y_pred = action

        return None, reward, done, {'y_pred': y_pred}


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

    def __init__(self, in_dim=1, hid_dim=128, out_dim=1):
        super(NET, self).__init__()

        self.fc1 = nn.Linear(in_dim, hid_dim)  # 6*6 from image dimension
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc_last = nn.Linear(hid_dim, out_dim)  # 0 or 1

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2(x))
        x = self.fc_last(x)  # with a regression output, so the activation of the last layer is linear.
        # x = torch.sigmoid(self.fc_last(x))
        return x


class RL():

    def __init__(self, in_dim=2, random_state=42):

        self.in_dim = in_dim
        self.GAMMA = 0.9
        self.BATCH_SIZE = 64

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        self.actions = ['normal', 'abnormal'] # ['pass', 'alarm', ''unsure(wait)''] #  'none'
        self.n_actions = len(self.actions)
        self.env = Environment()
        self.steps_done = 0
        self.random_state = 42

        self.policy_net = NET(in_dim=self.in_dim, out_dim=self.n_actions)
        self.target_net = NET(in_dim=self.in_dim, out_dim=self.n_actions)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.memory = ReplayMemory(2 * self.BATCH_SIZE)

    def optimize_model(self):
        loss = -1

        if len(self.memory) < self.BATCH_SIZE:
            return loss

        transitions = self.memory.sample(self.BATCH_SIZE)
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
            return loss
        non_final_next_states = torch.cat(v, axis=0)
        state_batch = torch.cat([v.view(1, -1) for v in batch.state], axis=0)
        action_batch = torch.cat([v.view(1, -1) for v in batch.action], axis=0)  # the index of actions
        reward_batch = torch.cat([v.view(1, -1) for v in batch.reward], axis=0) # R_(t+1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        self.policy_net.train() # in train mode
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # R_(t+1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch.squeeze(
            1)  # # R_(t+1) = R_t + gamma*(R_t+1)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.loss = loss

        return loss.item()

    def select_action(self, state, mode='train'):

        if mode == 'test':
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:  # mode == 'train'
            global steps_done
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                # randrange(self, start, stop=None, step=1, _int=int)
                # Choose a random item from range(start, stop[, step]).
                return torch.tensor([[random.randrange(0, stop=self.n_actions)]], device=device, dtype=torch.long)

    def train(self, X, y):
        # to make larger dataset
        n_repeats = 1
        # X, y = sklearn.utils.resample(X, y, n_samples=len(X)*n_repeats, random_state=self.random_state)
        X = X * n_repeats
        y = y * n_repeats

        X, y = sklearn.utils.shuffle(X, y, random_state=self.random_state)
        rewards_info = []
        pkts_info = []
        train_loss_info = []
        y_info = []
        for i_episode, (X_, y_) in enumerate(zip(X, y)):  # each flow is an episodes
            # Initialize the environment and state
            self.env.reset()  # Resets the environment and returns a random initial state.

            rewards_ = []
            # each packet is a state
            # state = (torch.Tensor(X_[0]), torch.Tensor([y_]))
            state = torch.Tensor(X_[0]).view(1, -1)
            pkt_0_time = 0
            i_pkt = 0
            tot_reward = 0  # tot_reward
            done = False
            loss = -1
            timeout = 10  # 10s
            stored_pkts = [X_[0]]
            while True and i_pkt < len(X_):
                action = self.select_action(state)  # normal: pass, abnormal: alarm
                # perform action and get next state and reward
                # wait_time = (pkt_i.time - pkt_0.time)
                wait_time = i_pkt - pkt_0_time
                _, reward, done, meta_info = self.env.step(action.item(), y=y_, wait_time=wait_time)
                tot_reward += reward   # reward is the instant reward after taking an action
                rewards_.append((reward, i_pkt))

                if not done:
                    if i_pkt + 1 < len(X_):
                        # next_state = (torch.Tensor(X_[i_pkt]), torch.Tensor([y_]))
                        stored_pkts.append(X_[i_pkt + 1])  # store the waited packets
                        # use the average as next state
                        next_state = np.mean(np.asarray([np.asarray(v) for v in stored_pkts]), axis=0)
                        next_state = torch.Tensor(next_state).view(1, -1)
                    else:
                        break
                else:
                    next_state = None  # final_state

                # Store the transition in memory
                self.memory.push(state, torch.tensor([action], dtype=int), next_state, torch.Tensor([reward]))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss = self.optimize_model()

                if done:
                    # episode_durations.append(i_pkt + 1)
                    # plot_durations()
                    # plot_rewards(rewards_)
                    # print(
                    #     f'i_episode: {i_episode}, tot_reward: {tot_reward}, i_pkt: {i_pkt}, rewards_: {rewards_}')
                    break
                i_pkt += 1
            if not done:
                # print(
                #     f'i_episode: {i_episode}, tot_reward: {tot_reward}, i_pkt: {i_pkt}, rewards_: {rewards_}, failed!')
                pass
            train_loss_info.append(loss)

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            y_info.append((y_, meta_info['y_pred']))
            rewards_info.append((tot_reward, rewards_))
            pkts_info.append((i_pkt + 1, len(X_) + 1))

        self.train_info = {'rewards_info': rewards_info, 'pkts_info': pkts_info, 'train_loss_info': train_loss_info,
                           'y_info': y_info}
        # print(f'info: {self.train_info}')
        # plot_info(info, is_show=True, title='train')
        # print(f'Complete')

        return self

    def test(self, X, y):
        """
        Let's evaluate the performance of our agent.
        We don't need to explore actions any further, so now the next action is always selected using the best Q-value:

        Parameters
        ----------
        X
        y

        Returns
        -------

        """
        # in eval mode
        self.policy_net.eval()
        self.target_net.eval()

        rewards_info = []
        pkts_info = []
        n_correct = 0
        y_info = []
        for i_episode, (X_, y_) in enumerate(zip(X, y)):  # each flow is an episodes

            # Initialize the environment and state
            self.env.reset()

            rewards_ = []
            # each packet is a state
            # state = (torch.Tensor(X_[0]), torch.Tensor([y_]))
            state = torch.Tensor(X_[0]).view(1, -1)
            pkt_0_time = 0
            i_pkt = 0
            tot_reward = 0
            done = False
            timeout = 10  # 10s
            stored_pkts = [X_[0]]
            while True:  # i_pkt < len(X_), only test the first 20 packets
                if i_pkt > 10:
                    print(f'faild, too much packets are needed!')
                    break
                action = self.select_action(state, mode='test')  # normal: pass, abnormal: alarm
                # perform action and get next state and reward
                # wait_time = (pkt_i.time - pkt_0.time)
                wait_time = i_pkt - pkt_0_time
                _, reward, done, meta_info = self.env.step(action.item(), y=y_, wait_time=wait_time)
                rewards_.append((reward, i_pkt))
                tot_reward += reward

                if not done:
                    if i_pkt + 1 < len(X_):
                        # next_state = (torch.Tensor(X_[i_pkt]), torch.Tensor([y_]))
                        # next_state = torch.Tensor(X_[i_pkt + 1]).view(1, -1)
                        stored_pkts.append(X_[i_pkt + 1])  # store the waited packets
                        next_state = np.mean(np.asarray([np.asarray(v) for v in stored_pkts]), axis=0)
                        next_state = torch.Tensor(next_state).view(1, -1)
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
                    # print(
                    #     f'i_episode: {i_episode}, tot_reward: {tot_reward}, i_pkt: {i_pkt}, rewards_: {rewards_}')
                    break
                i_pkt += 1
            if not done:
                print(
                    f'i_test_episode: {i_episode}, y:{y_}, tot_reward: {tot_reward}, i_pkt: {i_pkt}, rewards_: {rewards_}, failed!')
            else:
                n_correct += 1
            # # Update the target network, copying all weights and biases in DQN
            # if i_episode % TARGET_UPDATE == 0:
            #     target_net.load_state_dict(policy_net.state_dict())
            y_info.append((y_, meta_info['y_pred']))
            rewards_info.append((tot_reward, rewards_))
            pkts_info.append((i_pkt + 1, len(X_) + 1))
        self.test_info = {'rewards_info': rewards_info, 'pkts_info': pkts_info, 'y_info': y_info}
        # print(f'test_info: {self.test_info}')

        acc = n_correct / len(X)
        # print(f'test_acc: {acc}')
        y_true = [y_ for y_, y_pred_ in y_info]
        y_pred = [y_pred_ for y_, y_pred_ in y_info]
        cm = confusion_matrix(y_true, y_pred)
        print(f'cm: {cm}')
        print(f'acc: {accuracy_score(y_true, y_pred)}')

        return acc

    def show(self, info, title=''):
        plot_info(info, is_show=True, title=title)


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
        X_train, y_train, X_test, y_test = X, y, X, y

    else:
        # data_type = 'payload'  # header, header_payload
        # datasets = _generate_datasets()
        # for v_tup in datasets:
        #     (data_name, data_file), (X, y) = v_tup
        #     print(v_tup)
        X, y = load_flow_data(data_type='header', random_state=random_state)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random_state,
                                                            shuffle=True)
        X_train, y_train = balance_data(X_train, y_train)
        print(f'y_train: {Counter(y_train)}')
        X_test, y_test = balance_data(X_test, y_test)
        # X_train, y_train = X, y
        print(f'y_test: {Counter(y_test)}')
        in_dim = len(X_train[0][0])

    rl = RL(in_dim=in_dim)
    rl.train(X_train, y_train)
    rl.show(rl.train_info, title='train_phase')

    # evaulate
    title = '***evaulate on train set'
    print(f'\n\n{title}')
    train_acc = rl.test(X_train, y_train)
    rl.show(rl.test_info, title)
    title = '***evaulate on test set'
    print(f'\n\n{title}')
    test_acc = rl.test(X_test, y_test)
    rl.show(rl.test_info, title)

    print(f'***train_acc:{train_acc}, y_train: {Counter(y_train)}')
    print(f'***test_acc:{test_acc}, y_test: {Counter(y_test)}')


if __name__ == '__main__':
    main()
