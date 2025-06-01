import numpy as np
import logging
from typing import List
from gym.spaces import Discrete, MultiBinary
from monopoly_gym.player import Player
from monopoly_gym.state import State
from monopoly_gym.action import Action, ActionSpaceType, ActionManager, EndTurnAction

logger = logging.getLogger("MaskedRandomPlayer")

class MaskedRandomPlayer(Player):
    def __init__(
        self,
        name: str,
        mgn_code: str,
        starting_balance: int = 1500,
        action_space_type: ActionSpaceType = ActionSpaceType.HIERARCHICAL
    ):
        super().__init__(name, mgn_code, starting_balance)
        self.action_manager = ActionManager(action_space_type, include_send_message_action=False)
        self.action_space_type = action_space_type

    def decide_actions(self, game_state: State) -> List[Action]:
        action_mask = self.action_manager.to_action_mask(game_state)
        
        try:
            if self.action_space_type == ActionSpaceType.FLAT:
                return self._handle_flat_action(action_mask, game_state)
            else:
                return self._handle_hierarchical_action(action_mask, game_state)
        except Exception as e:
            logger.error(f"Error sampling action: {e}, defaulting to EndTurn")
            return [EndTurnAction(self)]

    def _handle_flat_action(self, action_mask: np.ndarray, game_state: State) -> List[Action]:
        valid_indices = np.where(action_mask)[0]
        if not valid_indices.size:
            return [EndTurnAction(self)]
        action_idx = np.random.choice(valid_indices)
        action = self.action_manager.decode_action(action_idx, game_state)
        return [action]

    def _handle_hierarchical_action(self, action_mask: dict, game_state: State) -> List[Action]:
        action_type_mask = action_mask['action_type']
        valid_action_indices = np.where(action_type_mask)[0]
        if not valid_action_indices.size:
            print("Defaulting to bad valid action indicies size")
            return [EndTurnAction(self)]
        
        chosen_action_idx = np.random.choice(valid_action_indices)
        action_cls = self.action_manager.action_classes[chosen_action_idx]
        parameters_mask = action_mask['parameters'][action_cls.__name__]
        
        parameters = {}
        for param_name, param_space in action_cls.hierarchical_parameters().items():
            if isinstance(param_space, Discrete):
                param_valid = parameters_mask.get(param_name, [True]*param_space.n)
                valid_options = np.where(param_valid)[0]
                if valid_options.size == 0:
                    parameters[param_name] = 0 
                else:
                    parameters[param_name] = np.random.choice(valid_options)
            elif isinstance(param_space, MultiBinary):
                n_bits = param_space.n
                param_valid = parameters_mask.get(param_name, [True]*n_bits)
                chosen_bits = []
                for i in range(n_bits):
                    if param_valid[i]:
                        chosen_bits.append(np.random.choice([0,1]))
                    else:
                        chosen_bits.append(0)
                parameters[param_name] = np.array(chosen_bits, dtype=int)

        
        action_dict = {
            "action_type": chosen_action_idx,
            "parameters": {action_cls.__name__: parameters}
        }
        action = self.action_manager.decode_action(action_dict, game_state)
        return [action]