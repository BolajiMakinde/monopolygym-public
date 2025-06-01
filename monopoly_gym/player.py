# monopoly_gym/player.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List
from monopoly_gym.action import Action
from monopoly_gym.state import State

from monopoly_gym.tile import Property


class Player(ABC):
    def __init__(self, name: str, mgn_code: str, starting_balance: int = 1500) -> None:
        self.name: str = name
        self.id = None
        self.mgn_code: str = mgn_code
        self.balance: int = starting_balance
        self.position: int = 0
        self.properties: List[Property] = []
        self.in_jail: bool = False
        self.jail_turns: int = 0
        self.jail_free_cards: int = 0
        self.is_bankrupt: bool = False

    @abstractmethod
    def decide_actions(self, game_state: State) -> List[Action]:
        """
        Decide the sequence of actions to perform during the turn.
        This method produces a vector of actions.
        """
        pass


    def __repr__(self):
        return (
            f"Player(name={self.name}, balance=${self.balance}, position={self.position}, "
            f"properties={len(self.properties)}, in_jail={self.in_jail})"
        )
    

    def __eq__(self, other: Any) -> bool:
        """
        Check for equality between two Player objects.
        Players are considered equal if their `name` and `mgn_code` are identical.
        """
        if not isinstance(other, Player):
            return False
        return self.name == other.name and self.mgn_code == other.mgn_code
    
    def __hash__(self):
        return hash(self.mgn_code)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mgn_code": self.mgn_code,
            "balance": self.balance,
            "position": self.position,
            "properties": [prop.to_dict() for prop in self.properties],
            "in_jail": self.in_jail,
            "jail_turns": self.jail_turns,
            "jail_free_cards": self.jail_free_cards,
            "is_bankrupt": self.is_bankrupt
        }
