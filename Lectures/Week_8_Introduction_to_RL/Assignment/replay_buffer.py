from collections import deque, namedtuple
from typing import Tuple
import random

import torch


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "terminated")
)


class ReplayMemory(object):
    """A simple replay buffer to store the transitions the agent observes."""

    def __init__(self, capacity: int) -> None:
        """Initialize the replay buffer

        Args:
            capacity (int): The maximum number of transitions that can be stored in the buffer,
            if the buffer is full, the oldest transitions will be removed to make space for new ones
        """
        self.memory = deque([], maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        terminated: torch.Tensor,
    ) -> None:
        """Add a new transition to the replay buffer

        Args:
            state (torch.Tensor): The starting state of the transition
            action (torch.Tensor): The action taken in the state
            reward (torch.Tensor): The reward received for the transition
            next_state (torch.Tensor): The resulting state of the transition
            terminated (torch.Tensor): Boolen whether the resulting state is a terminal state
        """
        self.memory.append(Transition(state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a mini-batch of transitions from the replay buffer

        Args:
            batch_size (int): Size of the mini-batch to sample

        Returns:
            Tuple[torch.Tensor, ...]: Tuple containing the mini-batch of states, actions, rewards, next_states and termination flags
        """
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        terminated_batch = torch.cat(batch.terminated)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            terminated_batch,
        )

    def __len__(self) -> int:
        """Return the current number of transitions stored in the replay buffer

        Returns:
            int: The current number of transitions stored in the replay buffer
        """
        return len(self.memory)
