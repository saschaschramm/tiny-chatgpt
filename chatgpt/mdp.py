import numpy as np
import networkx as nx
from chatgpt.mdp_nodes import StateNode
from chatgpt.mdp_nodes import ActionNode
from chatgpt.mdp_nodes import Node

import random
from typing import Optional


def _parity(list: list[int]) -> int:
    """Even parity = 0, Odd parity = 1"""
    return sum(list) % 2


class MDP:
    def __init__(self, root: StateNode, gamma: float=0, lam: float=0) -> None:
        self._root: StateNode = root
        self._root.add()
        self.graph = nx.Graph()
        self.gamma: float = gamma
        self.lam: float = lam


    def _add_state_node(self, child: StateNode) -> None:
        details: str = "".join([str(x) for x in child.state_details])
        node_infos = [
            f"$S_{child.t}={child.identifier}$",
            f"$context={details}$",
            f"$V_{child.t}={child.value:.3f}$",
            f"$R_{child.t}={child.reward:.2f}$",
            f"{child.debug}",
        ]

        self.graph.add_node(
            child.node_id, label="\n".join(node_infos), type="state", value=child.value
        )

    def get_networkx_graph(self):
        self._add_state_node(self._root)
        self._get_networkx_graph(self._root)
        return self.graph

    def _get_networkx_graph(self, node: Node) -> None:
        if node.children is None:
            return

        for child in node.children:
            if isinstance(node, StateNode):
                self.graph.add_node(
                    child.node_id,
                    label=f"$A_{child.t}={child.identifier}$\n$Q_{child.t}={child.value:.3f}$",
                    type="action",
                    prob=child.prob,
                    value=child.value,
                )
                label: str = f"$\\pi(A_{child.t}|S_{node.t})={child.prob}$"
                if child.prob is None:
                    self.graph.add_edge(
                        node.node_id, child.node_id, label=label, prob=0.0
                    )
                else:
                    self.graph.add_edge(
                        node.node_id, child.node_id, label=label, prob=child.prob
                    )

            elif isinstance(node, ActionNode):
                self._add_state_node(child)
                self.graph.add_edge(
                    node.node_id,
                    child.node_id,
                    label=f"$p={child.transition_prob}$",
                    prob=child.transition_prob,
                )

            self._get_networkx_graph(child)
        return

    def init(self) -> None:
        self._init(self._root)

    def _init(self, node: Node) -> None:
        # Add transition probabilities
        if isinstance(node, StateNode):
            state: int = node.identifier
            if node.parent is not None:
                node.state_details = node.parent.parent.state_details.copy()
                action: int = node.parent.identifier
                if action == state:
                    node.transition_prob = 1.0
                else:
                    node.transition_prob = 0.0
            node.state_details.append(state)

        # Add reward
        if node.children is None:
            context = node.state_details
            odd = _parity(context)
            node.reward = float(odd)
            return

        for child in node.children:
            self._init(child)
        return

    def temporal_difference(self, gamma: float) -> None:
        self._temporal_difference(self._root, gamma)

    def _sample_next_state(self, action_node: ActionNode) -> int:
        transition_probs: list[float] = []
        for next_state_nodes in action_node.children:
            transition_prob: float = next_state_nodes.transition_prob
            transition_probs.append(transition_prob)
        # action_node: a0
        # self: s0
        # next_state: s1
        # next_state.transition_prob: p(s1|s0,a0)
        # Sample next state (s1) with transition probabilities p(s1|s0,a0)
        next_state: int = random.choices([0, 1], weights=transition_probs)[0]
        return next_state

    def _sample_next_action(self, node: StateNode) -> Optional[ActionNode]:
        if node.children is None:
            raise ValueError("Node has no children")
        else:
            action_probs: list[Optional[float]] = [
                action_node.prob for action_node in node.children
            ]
            action: int = random.choices([0, 1], weights=action_probs)[0]
            action_node: Optional[ActionNode] = node.children[action]
            return action_node

    def _temporal_difference(self, node: StateNode, gamma: float) -> None:
        # Check if state is terminal
        if node.children is None:
            return

        action_node = self._sample_next_action(node)
        next_state: int = self._sample_next_state(action_node)
        next_state_node: Optional[StateNode] = action_node.children[next_state]
        self._temporal_difference(next_state_node, gamma)

        # The target value is the total return G
        node.target_value = (
            next_state_node.reward + gamma * next_state_node.target_value
        )

        # TD error = G0 - V(S0)
        node.error = node.target_value - node.value
        node.debug = f"$\\delta_{node.t}={node.error:.3f}$"

    def _update_state_values(self, node: Node, learning_rate: float) -> None:
        if node.children is None:
            return

        for child in node.children:
            self._update_state_values(child, learning_rate)

        if isinstance(node, StateNode):
            node.value += learning_rate * node.error

    def update_state_values(self, learning_rate: float) -> None:
        self._update_state_values(self._root, learning_rate)

    def gae(self) -> None:
        self.generalized_advantage_estimator = 0.0
        self._gae(self._root)

    def _gae(self, node: StateNode) -> None:
        if node.children is None:
            return

        action_node = self._sample_next_action(node)
        next_state: int = self._sample_next_state(action_node)
        next_state_node: Optional[StateNode] = action_node.children[next_state]
        self._gae(next_state_node)

        # δ = R1 + γ * V(S1) - V(S0)
        delta: float = (
            next_state_node.reward + self.gamma * next_state_node.value - node.value
        )

        # gae = δ + γ * λ * gae
        self.generalized_advantage_estimator = (
            delta + self.gamma * self.lam * self.generalized_advantage_estimator
        )

        target: float = self.generalized_advantage_estimator + node.value
        node.error = target - node.value

        node.debug = f"$GAE={self.generalized_advantage_estimator:.3f}$"

    def add_random_policy(self):
        return self._add_random_policy(self._root)

    def _add_random_policy(self, node: Node):
        if node.children is None:
            return

        if isinstance(node, ActionNode):
            node.prob = 0.5

        for node in node.children:
            self._add_random_policy(node)

    def add_deterministic_policy(self):
        return self._add_deterministic_policy(self._root)

    def _add_deterministic_policy(self, node):
        if node.children is None:
            return

        if isinstance(node, ActionNode):
            if node.identifier == 1:
                node.prob = 1.0
            else:
                node.prob = 0.0

        for node in node.children:
            self._add_deterministic_policy(node)

    def add_optimal_policy(self):
        return self._add_optimal_policy(self._root)

    def _add_optimal_policy(self, node):
        if node.children is None:
            return

        if isinstance(node, ActionNode):
            parent: StateNode = node.parent
            odd: int = _parity(parent.state_details + [node.identifier])
            if odd == 1:
                node.prob = 1.0
            else:
                node.prob = 0.0

        for node in node.children:
            self._add_optimal_policy(node)

    def policy_improvement(self) -> None:
        self._policy_improvement(self._root)

    def _policy_improvement(self, node: Node) -> None:
        if node.children is None:
            return

        if isinstance(node, StateNode):
            action_values: list[float] = []
            for action_node in node.children:
                self._policy_improvement(action_node)
                action_values.append(action_node.value)

            best_action: Optional[int] = None
            if len(action_values) > 0:
                best_action = int(np.argmax(action_values))
                for action_node in node.children:
                    if action_node.identifier == best_action:
                        action_node.prob = 1.0
                    else:
                        action_node.prob = 0.0

        elif isinstance(node, ActionNode):
            for state_node in node.children:
                self._policy_improvement(state_node)

    def update_values(self) -> None:
        return self._root.update_values(self.gamma)
