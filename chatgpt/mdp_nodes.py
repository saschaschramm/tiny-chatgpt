from typing import Optional
from chatgpt.node import Node


class ActionNode(Node):
    def __init__(self, timesteps: int, t: int, identifier: int, parent=None) -> None:
        self.parent = parent
        self.left_child: Optional[StateNode] = None
        self.right_child: Optional[StateNode] = None
        self.children: Optional[list[StateNode]] = None
        self.prob = 0.0
        self.value: float = 0.0
        self.debug: str = ""
        super().__init__(timesteps, t, identifier)

    def add_policy(self) -> None:
        if self.children is None:
            raise NotImplementedError

        self.prob = 0.5
        for state_node in self.children:
            state_node.add_policy()

    def add(self) -> None:
        self.left_child = StateNode(
            self.timesteps, identifier=0, t=self.t + 1, parent=self
        )
        self.right_child = StateNode(
            self.timesteps, identifier=1, t=self.t + 1, parent=self
        )
        self.children = [self.left_child, self.right_child]
        for child in self.children:
            child.add()

    def update_values(self, gamma: float) -> None:
        # Update state values with new policy
        if self.children is None:
            return
        self.value = 0.0
        for state_node in self.children:
            state_node.update_values(gamma)
            reward: float = state_node.reward
            transition_prob: float = state_node.transition_prob
            self.value += transition_prob * (reward + gamma * state_node.value)


class StateNode(Node):
    def __init__(self, timesteps: int, t: int, identifier: int, parent=None) -> None:
        self.parent = parent
        self.left_child: Optional[ActionNode] = None
        self.right_child: Optional[ActionNode] = None
        # Last state has no children
        self.children: Optional[list[ActionNode]] = None
        self.state_details: list[int] = []
        self.transition_prob = 0.0
        self.reward: float = 0.0
        self.value: float = 0.0
        self.debug: str = ""
        self.target_value: float = 0.0
        self.error: float = 0.0
        self.gamma = 0.9

        super().__init__(timesteps, t, identifier)

    def add_policy(self) -> None:
        if self.children is None:
            return

        for action_node in self.children:
            action_node.add_policy()

    def add(self) -> None:
        # Terminal state
        if self.t == self.timesteps:
            return

        self.left_child = ActionNode(
            self.timesteps, identifier=0, t=self.t, parent=self
        )
        self.right_child = ActionNode(
            self.timesteps, identifier=1, t=self.t, parent=self
        )
        self.children = [self.left_child, self.right_child]
        for state_node in self.children:
            state_node.add()

    def update_values(self, gamma: float) -> None:
        # Update action values with new policy
        if self.children is None:
            return
        self.value = 0.0
        for action_node in self.children:
            action_node.update_values(gamma)
            # Update state values
            self.value += action_node.prob * action_node.value
