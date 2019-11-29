import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def smoothenRewards(rewards):
    smoothed_reward = rewards[0]

    smoothened_rewards = [smoothed_reward]

    for i in range(1, len(rewards)):
        smoothed_reward = (0.99 * smoothed_reward) + (0.01 * rewards[i])
        smoothened_rewards.append(smoothed_reward)

    return smoothened_rewards

def readlines(lines):

    averageReward = []
    episode = []
    losses = []

    for i in range(0, len(lines), 2):
        line1 = lines[i].split(' ')
        line2 = lines[i+1].split(' ')

        # print(line1) # rewards
        # print(line2) # episode + loss

        # r = float((line1[5])[:-1])
        # r = '{0:.2f}'.format(r)

        r = line1[-1]
        r = float(r[:-2])
        e = int((line2[1])[:-1])
        loss = line2[-1]
        loss = float(loss[:-1]) * 100
        # loss = '{0:.4f}'.format(loss)

        averageReward.append(float(r))
        episode.append(e)
        losses.append(loss)


    return episode, averageReward, losses

if __name__ == '__main__':
    files = ['p1.txt', 'p2.txt', 'p3.txt', 'p4.txt']

    episodes = []
    averageRewards = []
    losses = []

    for file in files:
        f = open(file, 'r')
        lines = f.readlines()

        e, r, l = readlines(lines)

        episodes += e
        averageRewards += r
        losses += l

        f.close()

    # Code for plotting rewards graph
    averageRewards = smoothenRewards(averageRewards)
    plt.plot(episodes, averageRewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average rewards')
    plt.title('Smoothened rewards over training episodes')
    plt.xticks(np.arange(min(episodes), max(episodes) + 1, 10.0), rotation=90)
    plt.yticks(np.arange(min(averageRewards), max(averageRewards) + 1, 0.01))

    # Code for plotting loss graph
    # losses = smoothenRewards(losses)
    # plt.plot(episodes, losses)
    # plt.xlabel('Episodes')
    # plt.ylabel('Loss')
    # plt.title('Smoothened loss over training episodes')
    # plt.xticks(np.arange(min(episodes), max(episodes) + 1, 10.0), rotation=90)
    # plt.yticks(np.arange(min(losses), max(losses) + 1, 0.01))
    #

    ax = plt.axes()
    ax.locator_params(nbins=25)
    plt.legend()
    plt.show()
