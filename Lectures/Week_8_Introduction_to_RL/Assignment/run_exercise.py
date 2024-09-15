import gymnasium as gym
import torch

from agent import DQNAgent
from dqn import DQN
from replay_buffer import ReplayMemory
from trainer import Trainer


def main():
    # Initializing all components, setting all hyperparameter
    environment_name = "LunarLander-v2"
    train_environment = gym.make(environment_name, render_mode=None)
    # The evaluation environment has render mode set to human to visualize the agent after training
    evaluation_environment = gym.make(environment_name, render_mode="human")

    # Initializing the neural network
    deep_q_network = DQN(
        observation_dimensions=train_environment.observation_space.shape[0],
        number_of_actions=train_environment.action_space.n,
    )

    # Initializing the optimizer and telling it which parameters to optimize,
    # what objective is being optimized is determined by the gradients these parameters receive
    optimizer = torch.optim.AdamW(params=deep_q_network.parameters(), lr=1e-3)

    # Initializing the agent with a bunch of hyperparameters
    dqn_agent = DQNAgent(
        dqn=deep_q_network,
        optimizer=optimizer,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=1000,
        tau=0.005,
        action_space=train_environment.action_space,
        discount=0.99,
    )

    # Initializing the replay buffer
    replay_buffer = ReplayMemory(capacity=10000)

    # Initializing the trainer, that runs the training loop
    trainer = Trainer(
        train_environment=train_environment,
        evaluation_environment=evaluation_environment,
        agent=dqn_agent,
        memory=replay_buffer,
        batch_size=128,
    )

    # Running the training loop, If you think everything works as expected, you can increase the number of episodes to 200,
    # after that the agent should be able to solve the environment (most of the time)
    trainer.train(num_episodes=50)

    # Rendering the trained agent to see if training was successful
    trainer.evaluate(num_episodes=5)


if __name__ == "__main__":
    main()
