import torch
from sklearn.tree import DecisionTreeClassifier
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

actions = ['pass', 'alarm', 'none']

class Environment():

    def step(self, action):

        if action == 'pass':
            reward = 0
            done = False
        elif action == 'alarm':
            reward = -1
            done = False
        else:   # FP and FN
            reward = -20
            done = True

        return None, reward, done, None


class NET(nn.Module):

    def __init__(self, in_dim=1, out_dim=1):
        super(NET, self).__init__()

        self.fc1 = nn.Linear(in_dim, 256)  # 6*6 from image dimension
        self.fc2 = nn.Linear(256, out_dim)  # 0 or 1

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class Detector():

    def __init__(self, in_dim=2, out_dim=2):
        self.model = NET(in_dim=in_dim, out_dim=out_dim)
        self.epoches = 10

        # create your optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def fit(self, X, y):
        self.model.train()  # Train = True
        for epoch in range(self.epoches):
            # for X_, y_ in zip(X, y ):
            outs = self.model(X)
            outs = outs.max().flatten()  # choose the max one
            loss = self.criterion(outs, y)

            self.optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            self.optimizer.step()  # Does the weight update

        return self

    def predict_proba(self, X):
        self.model.eval()  # Train = False
        y = self.model(X)

        return y

#
# class RewardNet():
#     def __init__(self, in_dim=2, out_dim=2):
#         self.model = NET(in_dim=in_dim, out_dim=out_dim)
#         self.epoches = 10
#
#         # create your optimizer
#         self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
#         self.criterion = nn.MSELoss()
#
#     def fit(self, X, y):
#         self.model.train()  # Train = True
#         for epoch in range(self.epoches):
#             # for X_, y_ in zip(X, y ):
#             outs = self.model(X)
#             outs = outs.max().flatten()     # choose the max one
#             loss = self.criterion(outs, y)
#
#             self.optimizer.zero_grad()  # zero the gradient buffers
#             loss.backward()
#             self.optimizer.step()  # Does the weight update
#
#         return self
#
#     def predict_proba(self, X):
#         self.model.eval()  # Train = False
#         y = self.model(X)
#
#         return y

class Agent(Detector):

    def __init__(self, env):
        super(Agent, self).__init__()
        self.env = env

    def run(self, X, y):

        num_episodes = 50
        GAMMA = 0.6
        pkts_info = []
        rewards_info = []
        TARGET_UPDATE = 5
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            pkts_needed = []  # pkts_needed_per_flow
            rewards = []
            for X_, y_ in zip(X, y):
                # each time only process one flow: X_ (has multi-packets)
                rewards_ = []
                done = False
                i_ = len(X_)
                next_reward_pred = torch.Tensor([0])
                action = 'none'
                for i_, x_ in enumerate(X_):
                    x_ = torch.Tensor(x_)
                    y_ = torch.Tensor([y_])
                    if i_ == 0:
                        state = (x_, y_)
                        self.detector = Detector(in_dim=2, out_dim=len(actions))
                        self.detector.fit(x_, y_)
                        self.rewd_net = Detector(in_dim=2, out_dim=len(actions))      # predict the next reward based on the previous state and action
                        # v_ = torch.Tensor([1]) if action == 'pass' else torch.Tensor([0])
                        # X_rewd = torch.cat([x_, v_], axis=0).view((1, -1))
                        X_rewd = x_
                        y_rewd = y_
                        self.rewd_net.fit(X_rewd, y_rewd)
                    action = self.select_action(state)  # normal: pass, abnormal: alarm
                    _, reward, done, _ = self.env.step(action)
                    rewards_.append(reward)

                    next_reward_pred += reward * GAMMA
                    next_reward_true = self.rewd_net.predict_proba(X_rewd).max()
                    # loss = self.rewd_net.criterion(next_reward_true, next_reward_pred.flatten()) + torch.Tensor([i_*10])   # torch.Tensor([i_]) penilize number of packets
                    # Compute Huber loss
                    loss = F.smooth_l1_loss(next_reward_true, next_reward_pred.max())
                    # print(loss)
                    self.detector.optimizer.zero_grad()  # zero the gradient buffers
                    loss.backward()
                    self.detector.optimizer.step()

                    # # Observe new state
                    if not done:
                        # # Perform one step of the optimization (on the target network)
                        self.optimize_model(state)
                        state = (x_, y_)  # next_state
                        v_ = torch.Tensor([1]) if action == 'pass' else torch.Tensor([0])
                        # X_rewd = torch.cat([X_rewd, torch.cat([x_, v_], axis=0).view(1, -1)], axis=0)
                        # y_rewd = torch.cat([y_rewd, y_])
                        # X_rewd = torch.cat([x_, v_], axis=0).view((1, -1))
                        X_rewd = x_
                        y_rewd = y_
                    else:
                        state = None
                        pkts_needed.append((i_ + 1, len(X_)))
                        rewards.append(rewards_)
                        break

                if not done:
                    pkts_needed.append((i_ + 1, len(X_)))
                    rewards.append(rewards_)

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.rewd_net.model.load_state_dict(self.detector.model.state_dict())

            print(f'episode_{i_episode}, pkts_needed: {pkts_needed}, rewards: {rewards}')
            pkts_info.append(pkts_needed)
            rewards_info.append(rewards)

        self.info = {'pkts_info': pkts_info, 'rewards_info': rewards_info}

    def evaulate(self, X, y):
        pass

    def select_action(self, state):

        # global steps_done
        # sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #                 math.exp(-1. * steps_done / EPS_DECAY)
        # steps_done += 1
        # if sample > eps_threshold:
        if np.random.random() > 0.3:
            x, y = state
            y = y.item()
            y_pred = self.detector.predict_proba(x)
            # y_pred = y_pred.argmax().item()
            y_pred = y_pred.max().item()

            if y_pred > 0.8:
                y_pred = 1
            elif y < 0.2:
                y_pred = 0
            else:
                y_pred = 0.5

            if y == y_pred:  # abnormal: 1
                action = 'alarm'
            elif y == y_pred:  # normal: 0
                action = 'pass'
            else:
                action = 'none'  # FP and FN
        else:
            action = np.random.choice(actions)

        return action

    def optimize_model(self, state):
        X, y = state
        self.detector.fit(X, y)

        return self

    def show(self, info, is_show=True):
        fig, ax = plt.subplots(nrows=2, ncols=2)

        pkts_info = info['pkts_info']
        rewards_info = info['rewards_info']
        pkts_needed = []
        pkts = []
        for i_epoch, v_lst in enumerate(pkts_info):
            # v = [(2, 2), (3, 3), (1, 1), (4, 4)]
            pkts_needed.append([v for v, _ in v_lst])
            pkts.append([v for _, v in v_lst])

        rewards = []
        for i_epoch, v_lst in enumerate(rewards_info):
            # rewards: [[-1, -1], [-1, -1, -1], [1], [-1, -1, -1, 1]]
            rewards.append([sum(v) for v in v_lst])

        x = range(len(rewards))
        y = np.asarray(pkts_needed)
        yerr = np.std(y, axis=1)
        y = np.mean(y, axis=1)
        ax[0, 0].errorbar(x, y, yerr, ecolor='r', capsize=2, linestyle='-', marker='.', color='green',
                          markeredgecolor='green', markerfacecolor='green', label=f'pkts_needed',
                          alpha=0.9)  # marker='*',

        y = np.asarray(pkts)
        yerr = np.std(y, axis=1)
        y = np.mean(y, axis=1)
        ax[0, 0].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                          markeredgecolor='blue', markerfacecolor='blue', label=f'pkts', alpha=0.9)  # marker='*',
        ax[0, 0].legend(loc='upper right')
        ax[0, 0].set_xlabel('eposide')
        ax[0, 0].set_ylabel('pkts per flow')

        y = np.asarray(rewards)
        yerr = np.std(y, axis=1)
        y = np.mean(y, axis=1)
        ax[0, 1].errorbar(x, y, yerr, ecolor='m', capsize=2, linestyle='-', marker='.', color='blue',
                          markeredgecolor='blue', markerfacecolor='blue', label=f'rewards', alpha=0.9)  # marker='*',
        ax[0, 1].legend(loc='upper right')
        ax[0, 1].set_xlabel('eposide')
        ax[0, 1].set_ylabel('rewards')

        # # plt.xlim([0.0, 1.0])
        # if len(ylim) == 2:
        #     ax.set_ylim(ylim)  # [0.0, 1.05]
        # # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # # plt.xticks(x)
        # # plt.yticks(y)

        # ax.set_title(title)

        # fig.suptitle(title, fontsize=11)

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


def main():
    env = Environment()
    agt = Agent(env)

    X = [
        [[1, 1], [2, 1]],  # one row is one flow, which has multi-packets and each packet has 1500 bytes
        [[2, 2], [2, 2], [4, 4]],
        [[3, 3]],
        [[4, 4], [2, 4], [3, 3], [4, 4]],
    ]
    y = [0, 0, 1, 0]  # normal: 0, abnormal: 1

    agt.run(X, y)
    agt.show(agt.info, is_show=True)


if __name__ == '__main__':
    main()
