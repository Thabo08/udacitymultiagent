""" This script performs the training of the multi agent """

import datetime
import argparse
from unityagents import UnityEnvironment
from collections import deque
from common import *
import config
from support import Experience, ReplayBuffer, OUNoise
from agents import MultiDDPGAgent


def environment_settings(file_name="Tennis.app"):
    settings = {}
    env = UnityEnvironment(file_name=file_name)
    settings["env"] = env

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    settings["brain_name"] = brain_name

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    settings["num_agents"] = num_agents

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    settings["action_size"] = action_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('States look like:', states[0])
    print('States have length:', state_size)
    settings["state_size"] = state_size

    return settings


def step_tuple(env_info):
    """ Returns a tuple of next state, reward, and done when the agent steps through the environment based
        on the action taken
        :param env_info: Object holding information about the environment at a certain point
    """
    return env_info.vector_observations, env_info.rewards, env_info.local_done


def ddpg(agent: MultiDDPGAgent, env_settings: dict, num_episodes=2000, target=0.5, max_time_steps=500,
         saved_model="checkpoint.pth"):
    """ Train an agent using the DDPG algorithm

        :param env_settings: Settings of the environment
        :param agent: a continuous control agent
        :param num_episodes: the number of episodes to train the agent
        :param target: The average target score the agent needs to achieve for optimal performance
        :param max_time_steps: Maximum time steps per episode
        :param saved_model: The file path to save the model weights
    """
    now = datetime.datetime.now()
    print(now, "- Training a multi agent for max {} episodes. Target score to reach is {}".format(num_episodes, target))
    # collections to help keep track of the score
    scores_deque = deque(maxlen=100)
    scores = []
    stats = {"scores": [], "episodes": []}  # collects stats for plotting purposes
    mean_score = 0.

    env = env_settings["env"]
    brain_name = env_settings["brain_name"]
    num_agents = env_settings["num_agents"]

    for episode in range(1, num_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)

        for _ in range(max_time_steps):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = step_tuple(env_info)
            for idx in range(num_agents):
                agent.step(Experience(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx]))
            states = next_states
            score += rewards
            if np.any(dones):
                break
        max_score = max(score)
        scores_deque.append(max_score)
        scores.append(max_score)
        mean_score = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_score))

        stats["scores"].append(max_score)
        stats["episodes"].append(episode)

        if mean_score >= target:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, mean_score))
            print("Target score of {0} has been reached. Saving model to {1}".format(target, saved_model))
            torch.save(agent.local_actor_network().state_dict(), saved_model)
            break

    now = datetime.datetime.now()
    print(now, "- Finished training " + "successfully!" if mean_score >= target else "unsuccessfully!")
    return scores, stats


def test(agent: MultiDDPGAgent, env_settings, filename):
    print("Loading weights from {} to test the agent".format(filename))
    agent.local_actor_network().load_state_dict(torch.load(filename))
    env = env_settings["env"]
    brain_name = env_settings["brain_name"]
    num_agents = env_settings["num_agents"]

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations  # get the current state
    score = np.zeros(num_agents)  # initialize the score
    while True:
        action = agent.act(state, add_noise=False)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations
        # get the next state
        reward = env_info.rewards  # get the reward
        done = env_info.local_done  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if np.any(done):  # exit loop if episode finished
            break

    print("Score for {} agents: {}".format(num_agents, np.round(np.mean(score), 2)))

    env.close()


def plot(stats):
    scores = stats["scores"]
    episodes = stats["episodes"]
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.plot(episodes, scores)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.show()


def run_in_ide(train):
    env_settings = environment_settings()
    action_size = env_settings["action_size"]
    state_size = env_settings["state_size"]
    num_agents = env_settings["num_agents"]
    memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, random_seed=0)
    noise = OUNoise(action_size, 0)
    multi_agent = MultiDDPGAgent(state_size, action_size, num_agents, noise, memory)
    if train:
        scores, stats = ddpg(multi_agent, env_settings)
    else:
        test(multi_agent, env_settings, filename='checkpoint.pth')


def run_in_cmd():
    """Run in command line"""
    parser = argparse.ArgumentParser(description="""This script trains a agents to control racquets and play tennis with 
    the goal being to keep the tennis ball in play. It uses the Deep Deterministic Policy Gradient (DDPG) algorithm.""")
    parser.add_argument("agentFile", help="The file to load the agent(s)")
    parser.add_argument("--model", default="checkpoint.pth", help="Path where the trained model should be saved")
    parser.add_argument("--mode", default="train", choices=["train", "test"],
                        help="Mode describing whether to train or test")

    args = parser.parse_args()

    train = args.mode == "train"
    filename = args.model
    env_settings = environment_settings()
    action_size = env_settings["action_size"]
    state_size = env_settings["state_size"]
    num_agents = env_settings["num_agents"]
    memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, random_seed=0)
    noise = OUNoise(action_size, 0)
    multi_agent = MultiDDPGAgent(state_size, action_size, num_agents, noise, memory)

    if train:
        _, stats = ddpg(multi_agent, env_settings, saved_model=filename)
        plot(stats)
    else:
        test(multi_agent, env_settings, filename=filename)


if __name__ == '__main__':

    # uncomment the line below to run in IDE
    # run_in_ide(train=False)

    run_in_cmd()

