# monopoly_gym/env.py
#python3 -m monopoly_gym.env
import copy
import sys
from typing import Dict, List, Tuple, Union
import random
import logging
import coloredlogs
import pygame
import datetime 
import os

import gym
from monopoly_gym.renderer import Renderer
from monopoly_gym.action import Action, ActionSpaceType, HIERARCHICAL_ACTION_CLASSES, AuctionAction, AuctionBidAction, AuctionFoldAction, BankruptcyAction, EndTurnAction
from gym.spaces import Dict as GymDict, Discrete

from monopoly_gym.player import Player
from monopoly_gym.players.random_masked import MaskedRandomPlayer
from monopoly_gym.players.human_cli import HumanPlayer
from monopoly_gym.state import State
from monopoly_gym.tile import Chance, CommunityChest, Railroad, SpecialTile, SpecialTileType, Street, Property, Tax, Utility

MAX_JAIL_TURNS = 3
MAX_PLAYERS = 8
LEAVE_JAIL_FEE = 50
DEFAULT_MAX_TURNS = 10000

class MonopolyEnvironment(gym.Env):

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS, use_render: bool = True,
                 enable_general_log: bool = True, general_log_file: str = "monopoly_game.log",
                 enable_timestamped_log: bool = False, timestamped_log_dir: str = "logs"):

        # --- Configure Loggers ---
        self.env_logger = logging.getLogger("gym.env") 
        self.state_logger = logging.getLogger("gym.state")

        for logger_instance in [self.env_logger, self.state_logger]:
            for handler in list(logger_instance.handlers): 
                logger_instance.removeHandler(handler)
            logger_instance.propagate = False 

        log_formatter = logging.Formatter(
            "%(asctime)s %(name)-25s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        stream_h = logging.StreamHandler(sys.stdout)
        stream_h.setFormatter(log_formatter)
        stream_h.setLevel(logging.DEBUG)
        self.env_logger.addHandler(stream_h)
        self.state_logger.addHandler(stream_h)
        coloredlogs.install(level='DEBUG', logger=self.env_logger, fmt="%(asctime)s %(name)-25s %(levelname)-8s %(message)s", stream=sys.stdout)

        # General Log File (e.g., monopoly_game.log)
        if enable_general_log and general_log_file:
            general_file_h = logging.FileHandler(general_log_file, mode='a') # Append mode
            general_file_h.setFormatter(log_formatter)
            general_file_h.setLevel(logging.DEBUG) # Log all levels to this file
            self.env_logger.addHandler(general_file_h)
            self.state_logger.addHandler(general_file_h)

        # Timestamped Log File (optional, for individual game runs)
        if enable_timestamped_log and timestamped_log_dir:
            if not os.path.exists(timestamped_log_dir):
                os.makedirs(timestamped_log_dir, exist_ok=True)
            
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_log_filename = os.path.join(timestamped_log_dir, f"monopoly_game_{timestamp_str}.log")
            
            ts_file_h = logging.FileHandler(ts_log_filename, mode='w') 
            ts_file_h.setFormatter(log_formatter)
            ts_file_h.setLevel(logging.DEBUG)
            self.env_logger.addHandler(ts_file_h)
            self.state_logger.addHandler(ts_file_h)

        self.env_logger.setLevel(logging.DEBUG) 
        self.state_logger.setLevel(logging.INFO)

        self.state = State(max_turns=max_turns, logger=self.state_logger)

        if use_render:
            self.renderer = Renderer(name="MonopolyGym", state=self.state)
        else:
            self.renderer = None
        self.action_classes: List[Action] = HIERARCHICAL_ACTION_CLASSES
        self.use_render = use_render
        self.env_logger.info(f"MonopolyEnvironment initialized. Timestamped logs: {'Enabled' if enable_timestamped_log else 'Disabled'}")


    def render(self, mode='human'):
        """Render the current state."""
        if self.use_render == True:
            self.renderer.render()


    def shuffle_cards(self, deck):
        """Shuffle a card deck."""
        random.shuffle(deck)
        return deck

    def add_player(self, player: Player):
        if len(self.state.players) < MAX_PLAYERS:
            self.state.players.append(player)
        else:
            raise ValueError(f"Maximum number of players is {MAX_PLAYERS}.")

    def add_players(self, players: List[Player]):
        for player in players:
            self.add_player(player)


    def step(self, action: Action) -> Tuple[Union[State,GymDict,dict], float, bool, bool, dict]:
        if isinstance(action, Action):
            action.process(self.state)
        else:
            self.env_logger.error(f"Unknown action type: {action}")
        reward = None
        return self.state.to_dict(), reward, self.is_game_over(), {}

    def multistep(self, actions: List[Action]) -> Tuple[Union[State,GymDict,dict], float, bool, bool, dict]:
        for action in actions:
            self.step(action=action)
        reward = None
        return self.state.to_dict(), reward, self.is_game_over(), {}

    def validate_actions(self, actions: List[Action]) -> bool:
        # Clone the entire game state
        temp_state = copy.deepcopy(self.state)

        tile_map = {}
        for original_tile, cloned_tile in zip(self.state.board.board, temp_state.board.board):
            tile_map[original_tile] = cloned_tile

        for orig_action in actions:
            action_copy = copy.deepcopy(orig_action)

            if action_copy.player:
                cloned_player = next((p for p in temp_state.players 
                                    if p.name == action_copy.player.name 
                                        and p.mgn_code == action_copy.player.mgn_code),
                                    None)
                if cloned_player:
                    action_copy.player = cloned_player

            if hasattr(action_copy, "property"):
                if action_copy.property in tile_map:
                    action_copy.property = tile_map[action_copy.property]

            if hasattr(action_copy, "street"):
                if action_copy.street in tile_map:
                    action_copy.street = tile_map[action_copy.street]

            old_balance = action_copy.player.balance if action_copy.player else None
            action_copy.process(temp_state)
            new_balance = action_copy.player.balance if action_copy.player else None

        for p in temp_state.players:
            if p.balance < 0:
                self.env_logger.warning(f"Validation failed: {p.name} ended below $0.")
                sys.exit()
                return False

        return True


    def _multistep_validated_actions_util(self, player: Player) -> tuple:
        actions = player.decide_actions(self.state)
        self.multistep(actions=actions)


    def multistep_validated_actions(self):
        while True:
            player = self.state.current_player()
            actions = player.decide_actions(self.state)

            self.env_logger.info(f"[EXECUTE] Applying {[a.to_mgn() for a in actions]}")
            for action in actions:
                before = (action.player.balance, action.player.position, [p.name for p in action.player.properties])
                self.env_logger.info(f"[EXEC] → {action.to_mgn()} ({action.to_dict()})")
                action.process(self.state)
                after  = (action.player.balance, action.player.position, [p.name for p in action.player.properties])
                self.env_logger.info(f"[EXEC] {action.to_mgn()} ({action.to_dict()}) → bal {before[0]}->{after[0]}, pos {before[1]}->{after[1]}, props {before[2]}->{after[2]}")

            if any(isinstance(a, (EndTurnAction, BankruptcyAction)) for a in actions):
                self.env_logger.info(f"{player.name} ended their turn.")
                break

    def play(self):
        if not self.state.players:
            raise ValueError("No players added to the game")
        
        self.env_logger.info(f"Starting a new Monopoly game. Max turns: {self.state.max_turns}")
        self.env_logger.info(f"Players: {[p.name for p in self.state.players]}")

        if self.use_render:
            self.render(mode='human')
        while not self.is_game_over():
            if self.use_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        pygame.quit() # Ensure pygame quits
                        return # Exit play loop
            player = self.state.current_player()
            self.env_logger.info(f"[ENV] It's {player.name}'s turn, jail={player.in_jail}, rolled_this_turn={self.state.rolled_this_turn}, doubles_count={self.state.current_consecutive_doubles}")
            self.multistep_validated_actions()
            if self.use_render:
                self.render(mode='human')

            # Limit the frame rate
            if self.use_render:
                self.renderer.tick(framerate=30)

        self.env_logger.info(f"The game is over! {self.state.players[0].name} wins!")
        if self.use_render:
            pygame.quit()

    def is_game_over(self):
        return len(self.state.players) == 1  # Game ends when only one player remains

    def reset(self) -> dict:
        self.state.reset()
        self.env_logger.info("MonopolyEnvironment state has been reset.")
        return self.state.to_dict()

    def close(self):
        self.env_logger.info("Closing MonopolyEnvironment.")
        for handler in list(self.env_logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
            self.env_logger.removeHandler(handler)
        for handler in list(self.state_logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
            self.state_logger.removeHandler(handler)
        if self.use_render and self.renderer and self.renderer.running: # Check if renderer is running
             pygame.quit()


if __name__ == "__main__":
    max_turns = 300  # Change this value to set the maximum number of turns
    game = MonopolyEnvironment(
        max_turns=max_turns,
        use_render=True, # Set to False for faster non-UI runs
        enable_general_log=True,
        general_log_file="monopoly_game_main.log", # Can rename the general log
        enable_timestamped_log=True, 
        timestamped_log_dir="game_logs" # Directory for timestamped game logs
    )


    # Add players
    players = [
        MaskedRandomPlayer(name="AI 1", mgn_code="A1", action_space_type=ActionSpaceType.HIERARCHICAL),
        MaskedRandomPlayer(name="AI 2", mgn_code="A2", action_space_type=ActionSpaceType.HIERARCHICAL),
        MaskedRandomPlayer(name="AI 3", mgn_code="A3", action_space_type=ActionSpaceType.HIERARCHICAL),
    ]
    for player in players:
        game.add_player(player=player)

    # Start the game
    game.env_logger.info("Starting the Monopoly game from __main__!")
    try:
        game.play()
    except Exception as e:
        game.env_logger.exception("An error occurred during game play:")
    finally:
        game.close() 