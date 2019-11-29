import gym
from model import Model
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

verbosity = 2  # 1 = log information, 2 = render scenes
OBSERVATIONS_SIZE = 6400  # Preprocessed observation size

# From Andrej's code
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discountRewards(rewards, discountFactor):
    """
    Gives a discounted reward to each of the frame, s.t. each frame has a scalar for gradient calculation
    :param rewards: list
    :param discountFactor: float32
    :return:
    """
    segments = []
    currentSeg = []

    # Perform segmentation on batch of rewards into different games
    for r in rewards:
        currentSeg.append(r)
        if r != 0:
            segments.append(currentSeg)
            currentSeg = []

    if currentSeg != []:
        segments.append(currentSeg)

    # Perform reward discount on each frame
    output = []
    for seg in segments:
        # print('before discount: {}'.format(seg))
        reward_sum = sum(seg)
        discount = 1
        discounted_rewards = [0.] * len(seg)

        for i in range(len(seg)):
            # print(discount * reward_sum)
            discount = discountFactor ** (len(seg) - i - 1)
            discounted_rewards[i] = discount * reward_sum
            # discount *= discountFactor

        # print(discounted_rewards)
        output += discounted_rewards
        # print('after discount: {}'.format(discounted_rewards))

    output = np.array(output)
    # print('output: {}'.format(output))
    return output

def mapAction(action, gymSpace=True):
    """
    Maps gym space action to model space action or vice versa
    :param action: int32
    :param gymSpace: Indication for which action space 'action' is in
    :return:
    """
    if gymSpace:
        if action == 2:
            return 1
        if action == 3:
            return 0
    else:
        if action == 1:
            return 2
        else:
            return 3


def train(model: Model):
    model.training = True

    episodes = 3000
    log_every_episode = 5
    save_every_episode = 50

    log_file = './training_log.txt'
    f = open(log_file, 'a+')

    reward_history = []
    reward_averaged = []

    for ep in range(episodes):
        ob = model.env.reset()
        ob = prepro(ob)
        new_ob, _, _, _ = model.env.step(model.env.action_space.sample())
        new_ob = prepro(new_ob)
        done = False

        episode_states = []
        episode_actions = []
        episode_rewards = []

        episode_rewards_total = 0.

        while not done:
            if verbosity >= 2:
                model.env.render()

            ob_delta = new_ob - ob
            ob = new_ob

            action = model.return_action(ob_delta)
            new_ob, reward, done, info = model.env.step(action)
            new_ob = prepro(new_ob)

            episode_states.append(ob_delta)
            episode_actions.append(mapAction(action, gymSpace=True))
            episode_rewards.append(reward)

            episode_rewards_total += reward

        # On completion of one episode

        # Rewards discount and standardization, refer back to Andrej's article for more detail on this
        discounted_rewards = discountRewards(episode_rewards, discountFactor=0.99)
        rewards_mean = np.mean(discounted_rewards)
        rewards_stdd = np.std(discounted_rewards)

        discounted_rewards -= rewards_mean
        discounted_rewards /= rewards_stdd

        model.current_step += 1

        episode_states = np.vstack(episode_states)
        episode_actions = np.vstack(episode_actions)
        discounted_rewards = np.vstack(discounted_rewards)

        feed_dict = {
            model.states: episode_states,
            model.actions: episode_actions,
            model.rewards: discounted_rewards
        }

        _, loss = model.sess.run(
            [model.train_op, model.loss],
            feed_dict=feed_dict
        )

        reward_history.append(episode_rewards_total)
        reward_averaged.append(np.mean(reward_history[-10:]))  # obtain mean of 10 most recent games

        if verbosity >= 1 and ep % log_every_episode == 0:
            reward_info = "Max reward: {}, Average reward: {}, accumulated_reward: {}\n".format(np.max(reward_history),
                                                                                                np.mean(reward_history),
                                                                                                episode_rewards_total)
            loss_info = "Episode: {}, log loss: {}\n".format(model.current_step, loss)

            print(loss_info)
            print(reward_info)
            f.write(reward_info)
            f.write(loss_info)

        if ep % save_every_episode == 0:
            model.save_model(model.current_step)

    model.save_model(model.current_step)
    f.close()

def inference(model: Model):
    model.training = False

    while True:
        done = False
        ob = model.env.reset()
        ob = prepro(ob)

        new_ob, _, _, _ = model.env.step(model.env.action_space.sample())
        new_ob = prepro(new_ob)

        while not done:
            model.env.render()

            ob_delta = new_ob - ob
            ob = new_ob

            action = model.return_action(ob_delta)
            new_ob, _, done, _ = model.env.step(action)
            new_ob = prepro(new_ob)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Insufficient command line arguments provided, please choose model mode')
        sys.exit()
        
    mode = sys.argv[1]

    env = gym.make('Pong-v0')

    rlModel = Model(OBSERVATIONS_SIZE, env)

    with rlModel.sess.as_default():
        if mode == 'train':
            train(rlModel)
        elif mode == 'inference':
            inference(rlModel)
        else:
            print('Unrecognized input mode "{}", please choose from [train, inference]'.format(mode))

    env.close()

