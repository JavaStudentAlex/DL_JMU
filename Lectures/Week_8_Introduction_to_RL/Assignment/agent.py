import copy
import math
import random
import torch
import gymnasium as gym
from torch import nn


class DQNAgent:
    def __init__(
        self,
        dqn: torch.nn.Module,
        optimizer: torch.nn.Module,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        action_space: gym.spaces.Space,
        discount: float,
        tau: float,
    ) -> None:

        self.dqn = dqn
        # Our target_net starts of as a copy of our online net, but as the online net changes
        # the target_net is only updated in a delayed manner and therefore they will differ
        self.target_net = copy.deepcopy(self.dqn)

        self.optimizer = optimizer

        # We use an decaying epsilon such that there is more exploration in the beginning
        # and more exploitation as time goes an
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.discount = discount
        # tau is the percentage by with we update our target_net towards our online_net in each update step
        self.tau = tau
        self.action_space = action_space

        # Same thing as the Huber loss with delta equal to 1
        self.criterion = nn.SmoothL1Loss()

        # We use this to calculate the current epsilon
        self.steps_done = 0

    def getPolicy(self, state: torch.tensor) -> int:
        """The the best action according to the current Q-Value estimates

        Args:
            state (torch.tensor): The state for which the action should be determined

        Returns:
            int: id of the choosen action
        """
        with torch.no_grad():
            return self.dqn(state).argmax().unsqueeze(0).unsqueeze(0)

    def getAction(self, state: torch.tensor) -> int:
        """Get an action according to an epsilon greedy policy, this means that with a probability of epsilon
        a random action is chosen and with a probability of 1-epsilon the best action according to the current Q-Value
        Here the epsilon is decaying over time to shift from more exploration to more exploitation

        Args:
            state (torch.tensor): The state for which the action should be determined

        Returns:
            int: id of the choosen action
        """
        sample = random.random()

        # calculate decaying epsilon
        eps_threshold = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_decay)

        self.steps_done += 1
        if sample > eps_threshold:
            return self.getPolicy(state=state)
        else:
            return torch.tensor([[self.action_space.sample()]], dtype=torch.long)

    def update(
        self,
        state_batch: torch.tensor,
        action_batch: torch.tensor,
        reward_batch: torch.tensor,
        next_state_batch: torch.tensor,
        terminated_batch: torch.tensor,
    ) -> None:
        """Perform a single step of the optimization to the DQN

        Args:
            state_batch (torch.tensor): minibatch of starting states of the transitions
            action_batch (torch.tensor): minibatch of actions of the transitions
            reward_batch (torch.tensor): minibatch of rewards of the transitions
            next_state_batch (torch.tensor): minibatch of the resulting states of the transitions
            terminated_batch (torch.tensor): minibatch of termination flags of the resulting states of the transitions
        """

        # This might be useful, ~ simply inverts booleans
        not_terminated = ~terminated_batch

        # TODO calculate the current estimates for the given state action paris
        # the .gather() function from pytorch might be very useful, remember that the output of the network is a Q-Value for each action
        # thus the output of self.dqn(state_batch) is a tensor of shape (batch_size, num_actions)
        prediction = ...

        # preparing a tensor of zeros with the same shape as the prediction tensor for out next state estimates which we need to calculate the target value
        next_state_values = torch.zeros_like(prediction)

        # This flag indicates to pytorch to not track gradients within the scope, this saves computation
        # and is also necessary as we want to implement the semi-gradient variant (we only want to update the estimate towards the target, not the target towards the estimate)
        with torch.no_grad():
            # TODO calculate the estimates for the next state (action) pairs, deep neural networks are computationally expensive (especially when running on the cpu)
            # try not to pass things to the network you don't really need, the not_terminated booleans should help you with this.
            # Also you might need to add a Batch dimension again back to your result like (batch_size, 1)
            ...

        # TODO using the next_state values calculate the target value

        # TODO calculate the loss using the given self.criterion

        # TODO clear all previous gradients from the optimizer

        # TODO Let pytorch calculate all gradients of the parameters with respect to the loss

        # TODO for stability reasons clip all gradients to an absolute value of 10

        # TODO Finally let the optimizer perform an optimization step based on the gradients

        # Update the target network
        self._update_target_net()

    def _update_target_net(self) -> None:
        """Update all parameters of the target network towards the online network by a factor of tau"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.dqn.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
