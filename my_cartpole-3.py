from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.optim import RMSprop
import numpy as np
import gym
import matplotlib.pyplot as plt

@dataclass
class hyp:
    ACTOR_LR: float = 0.001
    CRITIC_LR: float = 0.01
    NUM_HIDDEN_NEURONS: int = 128
    NUM_AGENTS: int = 4
    NUM_EPOCHS: int = 100
    DISCOUNT: float = 0.99
    RMSPROP_ALPHA: float = 0.9
    RMSPROP_EPS: float = 1e-10
    BATCH_SIZE: int = 100
    T_MAX: int = 5
    SMOOTHING_AMT: int = 5 # BATCH_SIZE // 10

@dataclass
class cons:
    NUM_IN_NEURONS: int = 4
    NUM_ACTS: int = 2
    DEVICE: str = 'cpu'
    ENV: str = 'CartPole-v0'
    PRINT_EVERY: int = 2

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.actor_layer_1 = nn.Linear(cons.NUM_IN_NEURONS, hyp.NUM_HIDDEN_NEURONS)
        self.actor_layer_2 = nn.Linear(hyp.NUM_HIDDEN_NEURONS, cons.NUM_ACTS)

    def forward(self, x):
        x = F.relu(self.actor_layer_1(x))
        action_probs = F.softmax(self.actor_layer_2(x), dim=1)
        return action_probs

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic_layer_1 = nn.Linear(cons.NUM_IN_NEURONS, hyp.NUM_HIDDEN_NEURONS)
        self.critic_layer_2 = nn.Linear(hyp.NUM_HIDDEN_NEURONS, 1)

    def forward(self, x):
        x = F.relu(self.critic_layer_1(x))
        value = self.critic_layer_2(x)
        return value

def central_agent(parameter_queues, experience_queues):
    actor = Actor()
    actor_criterion = RMSprop(actor.parameters(), lr=hyp.ACTOR_LR, alpha=hyp.RMSPROP_ALPHA, eps=hyp.RMSPROP_EPS)
    actor_criterion.zero_grad()
    critic = Critic()
    critic_criterion = RMSprop(critic.parameters(), lr=hyp.CRITIC_LR, alpha=hyp.RMSPROP_ALPHA, eps=hyp.RMSPROP_EPS)
    critic_criterion.zero_grad()
    critic_loss_fn = nn.MSELoss()

    for epoch in range(hyp.NUM_EPOCHS):
        actor_parameters = actor.parameters()

        for i in range(hyp.NUM_AGENTS):
            parameter_queues[i].put(list(actor_parameters))

            # to use CUDA with multiprocessing you need to use the spawn method which I have no clue about
            states, actions, rewards, terminal = experience_queues[i].get()
            states = torch.tensor(np.array(states))
            actions = torch.tensor(np.array(actions))
            rewards = torch.tensor(np.array(rewards))
            returns = torch.zeros(rewards.shape)

            returns[-1] = rewards[-1]
            for t in reversed(range(rewards.shape[0] - 1)):
                returns[t] = rewards[t] + hyp.DISCOUNT * returns[t + 1]
            with torch.no_grad():
                values = critic.forward(states).squeeze()
                advantage = returns - values

            probs = actor.forward(states)
            dist = Categorical(probs)
            actor_loss = torch.sum(dist.log_prob(actions) * (- advantage))
            actor_loss.backward()

            critic_loss = critic_loss_fn(returns, critic.forward(states).squeeze())
            critic_loss.backward()

        actor_criterion.step()
        actor_criterion.zero_grad()
        critic_criterion.step()
        critic_criterion.zero_grad()

def agent(i, agent_parameter_queue, agent_experience_queue):
    actor = Actor()

    env = gym.make(cons.ENV)
    scores = []

    for epoch in range(hyp.NUM_EPOCHS):
        global_actor_params = agent_parameter_queue.get()
        for target, source in zip(actor.parameters(), global_actor_params):
            target.data.copy_(source.data)
        state = env.reset()
        states, actions, rewards = [state], [], []

        done = False
        time_step = 0
        score = 0
        steps = 0
        while not done and steps < 250:
            with torch.no_grad():
                probs = actor.forward(torch.tensor(np.array(states)))
                dist = Categorical(probs)
                action = dist.sample().numpy()[0]

            next_state, reward, done, info = env.step(action)
            score += reward 

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            time_step += 1
            steps += 1
            
            if len(states) >= hyp.BATCH_SIZE or time_step >= hyp.T_MAX:
                agent_experience_queue.put([states[1: ], actions, rewards, done])
                states, actions, rewards = [state], [], []
                time_step = 0
        scores.append(score)
        if epoch + 1 % cons.PRINT_EVERY == 0:
            print('epoch', epoch)
    scores = moving_average(scores, n=hyp.SMOOTHING_AMT)
    plot(str(i), scores)

def moving_average(a, n=3) :
    ret = np.cumsum(np.array(a), dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot(name, scores):
    path = r'/mnt/windows_d/LibraryOfBabel/Images/Miscellaneous/thesis/my-cartpole-3-' + name + '.png'
    plt.plot([i for i in range(1, len(scores) + 1)], scores)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.title('worker ' + name)
    plt.savefig(path, bbox_inches='tight')


if __name__ == '__main__':
    parameter_queues = [mp.Queue(1) for i in range(hyp.NUM_AGENTS)]
    experience_queues = [mp.Queue(1) for i in range(hyp.NUM_AGENTS)]
    
    coordinator = mp.Process(target=central_agent, args=(parameter_queues, experience_queues))
    coordinator.start()

    agents = [mp.Process(target=agent, args=(i, parameter_queues[i], experience_queues[i])) for i in range(hyp.NUM_AGENTS)]
    for process in agents:
        process.start()

    coordinator.join()
    for process in agents:
        process.join()
