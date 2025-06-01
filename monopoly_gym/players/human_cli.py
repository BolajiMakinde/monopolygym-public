import logging
import numpy as np
from typing import List, Dict, Union
from gym.spaces import Discrete, MultiBinary

from monopoly_gym.player import Player
from monopoly_gym.state import State
from monopoly_gym.action import (
    Action, ActionSpaceType, ActionManager, EndTurnAction, SendMessageAction)

logger = logging.getLogger("HumanPlayer")


class HumanPlayer(Player):
    def __init__(self, name: str, mgn_code: str, starting_balance: int = 1500):
        super().__init__(name, mgn_code, starting_balance)
        self.action_manager = ActionManager(ActionSpaceType.HIERARCHICAL)
        self.action_space_type = ActionSpaceType.HIERARCHICAL

    def decide_actions(self, game_state: State) -> List[Action]:
        print("\n" + "="*40)
        print(f"It's your turn, {self.name} ({self.mgn_code})!")
        print(f"Balance: ${self.balance}")
        print(
            f"Position: {game_state.board.board[self.position].name} ({self.position})")
        print(
            f"In Jail: {self.in_jail} (Turns: {self.jail_turns}, Cards: {self.jail_free_cards})")
        print("Properties:")
        if not self.properties:
            print("  None")
        else:
            for prop in sorted(self.properties, key=lambda p: p.index):
                mortgaged = "(M)" if prop.is_mortgaged else ""
                houses = ""
                if hasattr(prop, 'houses'):
                    houses = f" (H:{prop.houses}, HTL:{prop.hotels})"
                print(f"  - {prop.name} ({prop.index}) {mortgaged}{houses}")
        print("="*40)
        action_mask = self.action_manager.to_action_mask(game_state)
        action_type_mask = action_mask['action_type']
        valid_action_indices = np.where(action_type_mask)[0]

        if not valid_action_indices.size:
            print("No valid actions available. Ending turn.")
            return [EndTurnAction(self)]
        print("Available Actions:")
        valid_actions_map = {}
        for i, idx in enumerate(valid_action_indices):
            action_cls = self.action_manager.action_classes[idx]
            print(f"  {i+1}: {action_cls.__name__}")
            valid_actions_map[i + 1] = (idx, action_cls)
        chosen_action_cls = None
        chosen_action_idx = -1
        while chosen_action_cls is None:
            try:
                choice = int(input("Choose an action number: "))
                if choice in valid_actions_map:
                    chosen_action_idx, chosen_action_cls = valid_actions_map[choice]
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a number.")
        parameters = {}
        parameters_mask = action_mask['parameters'][chosen_action_cls.__name__]
        param_spaces = chosen_action_cls.hierarchical_parameters()

        print(f"Selected: {chosen_action_cls.__name__}")
        if chosen_action_cls.__name__ == "SendMessageAction":
            print("\nWho do you want to message?")
            print("  0: ALL")
            for i, p in enumerate(game_state.players, start=1):
                print(f"  {i}: {p.name}")
            while True:
                try:
                    r = int(input("Recipient number: "))
                    if 0 <= r <= len(game_state.players):
                        break
                except ValueError:
                    pass
                print("Invalid choice.")

            recipient = None if r == 0 else game_state.players[r-1]

            msg = input("Enter your message: ")
            return [SendMessageAction(self, msg, recipient)]
        for param_name, param_space in param_spaces.items():
            if isinstance(param_space, Discrete):
                param_valid_mask = parameters_mask.get(
                    param_name, [True]*param_space.n)
                valid_options = np.where(param_valid_mask)[0]

                if valid_options.size == 0:
                    print(f"  No valid options for {param_name}. Defaulting.")
                    parameters[param_name] = 0 
                    continue
                if valid_options.size == 1 and param_space.n > 1:
                    parameters[param_name] = valid_options[0]
                    print(f"  Auto-selected {param_name}: {valid_options[0]}")
                    continue
                if param_space.n == 1:
                    parameters[param_name] = 0
                    continue
                display_options = []
                if param_name == "trade_partner":
                    possible_responders = [
                        p for p in game_state.players if p != self]
                    for i, option_idx in enumerate(valid_options):
                        if option_idx < len(possible_responders):
                            display_options.append(
                                f"{i+1}: {possible_responders[option_idx].name}")
                        else:
                            display_options.append(
                                f"{i+1}: Invalid Partner Index {option_idx}")
                    prompt_text = f"Choose {param_name}:\n  " + \
                        "\n  ".join(display_options) + "\nYour choice: "
                elif param_name in ["property", "street"]:
                    prop_attr = "properties" if param_name == "property" else "streets"
                    prop_list = getattr(game_state.board, prop_attr)
                    for i, option_idx in enumerate(valid_options):
                        if 0 <= option_idx < len(prop_list):
                            display_options.append(
                                f"{i+1}: {prop_list[option_idx].name} ({prop_list[option_idx].index})")
                        else:
                            display_options.append(
                                f"{i+1}: Invalid Index {option_idx}")
                    prompt_text = f"Choose {param_name}:\n  " + \
                        "\n  ".join(display_options) + "\nYour choice: "
                elif param_name == "quantity":
                    for i, option_idx in enumerate(valid_options):
                        display_options.append(
                            f"{i+1}: {option_idx+1} building(s)")
                    prompt_text = f"Choose {param_name}:\n  " + \
                        "\n  ".join(display_options) + "\nYour choice: "
                elif param_name in ["cash_offered", "cash_asking", "bid_amount"]:
                    min_val = valid_options[0]
                    max_val = valid_options[-1]
                    prompt_text = f"Enter {param_name} (between {min_val} and {max_val}): "
                    while True:
                        try:
                            value = int(input(prompt_text))
                            if min_val <= value <= max_val:
                                parameters[param_name] = value
                                break
                            else:
                                print(
                                    f"Value must be between {min_val} and {max_val}.")
                        except ValueError:
                            print("Invalid number.")
                    continue
                else:
                    for i, option_idx in enumerate(valid_options):
                        display_options.append(f"{i+1}: {option_idx}")
                    prompt_text = f"Choose {param_name}:\n  " + \
                        "\n  ".join(display_options) + "\nYour choice: "
                chosen_param_val = -1
                while chosen_param_val == -1:
                    try:
                        param_choice = int(input(prompt_text))
                        if 1 <= param_choice <= len(valid_options):
                            chosen_param_val = valid_options[param_choice - 1]
                            parameters[param_name] = chosen_param_val
                        else:
                            print("Invalid choice.")
                    except ValueError:
                        print("Please enter a number.")

            elif isinstance(param_space, MultiBinary):
                param_valid_mask = parameters_mask.get(
                    param_name, [True]*param_space.n)
                valid_indices = [i for i, valid in enumerate(
                    param_valid_mask) if valid]
                print(
                    f"Select {param_name} (MultiBinary - enter indices separated by space, e.g., '1 3 5'):")
                display_map = {}
                prop_list = []
                if param_name == "properties_offered":
                    prop_list = [p for p in self.properties]
                elif param_name == "properties_asking":
                    prop_list = game_state.board.properties

                for i, idx in enumerate(valid_indices):
                    prop_name = f"Index {idx}"
                    actual_prop = next(
                        (p for p in prop_list if p.property_idx == idx), None)
                    if actual_prop:
                        prop_name = f"{actual_prop.name} ({actual_prop.index})"
                    display_map[i+1] = idx
                    print(f"  {i+1}: {prop_name}")

                chosen_bits = np.zeros(param_space.n, dtype=int)
                while True:
                    try:
                        raw_input = input(
                            "Enter choices (space-separated numbers, or ENTER for none): ")
                        if not raw_input.strip():
                            break
                        selected_choices = [int(x) for x in raw_input.split()]
                        valid_input = True
                        selected_indices = []
                        for choice in selected_choices:
                            if choice in display_map:
                                selected_indices.append(display_map[choice])
                            else:
                                print(f"Invalid choice: {choice}")
                                valid_input = False
                                break
                        if valid_input:
                            for idx in selected_indices:
                                chosen_bits[idx] = 1
                            break 
                    except ValueError:
                        print(
                            "Invalid input. Please enter numbers separated by spaces.")
                parameters[param_name] = chosen_bits
        action_dict = {
            "action_type": chosen_action_idx,
            "parameters": {chosen_action_cls.__name__: parameters}
        }
        action = self.action_manager.decode_action(action_dict, game_state)
        return [action] 
