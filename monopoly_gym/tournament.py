# monopoly_gym/tournament.py

import os
import uuid
import datetime
import json
import csv
import logging
import itertools
import random
import time
import sys
import io
import contextlib
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional, Type, Callable

from monopoly_gym.player import Player
from monopoly_gym.env import MonopolyEnvironment
from monopoly_gym.state import State
from monopoly_gym.action import Action

@dataclass
class TournamentConfig:
    name: str = "MonopolyTournament"
    num_players_per_game_range: Tuple[int, int] = (2, 4)
    num_matches_per_pairing: int = 1
    pairing_strategy: str = "all_vs_all" # "random_groups", "all_vs_all"
    num_random_games_if_strategy_random: int = 10
    shuffle_turn_order_in_redundant_matches: bool = True
    max_turns_per_game: int = 1000
    agent_timeout_seconds: float = 60.0

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentStats:
    agent_id: str
    agent_name: str
    agent_type: str
    games_played: int = 0
    wins: int = 0
    total_turns_in_wins: int = 0
    total_turns_in_losses: int = 0
    total_duration_in_wins_s: float = 0.0
    total_duration_in_losses_s: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games_played if self.games_played > 0 else 0.0

    @property
    def avg_turns_in_wins(self) -> float:
        return self.total_turns_in_wins / self.wins if self.wins > 0 else 0.0

    @property
    def avg_turns_in_losses(self) -> float:
        losses = self.games_played - self.wins
        return self.total_turns_in_losses / losses if losses > 0 else 0.0

    @property
    def avg_duration_in_wins_s(self) -> float:
        return self.total_duration_in_wins_s / self.wins if self.wins > 0 else 0.0

    @property
    def avg_duration_in_losses_s(self) -> float:
        losses = self.games_played - self.wins
        return self.total_duration_in_losses_s / losses if losses > 0 else 0.0

    def to_dict(self):
        data = asdict(self)

        data['win_rate'] = self.win_rate
        data['avg_turns_in_wins'] = self.avg_turns_in_wins
        data['avg_turns_in_losses'] = self.avg_turns_in_losses
        data['avg_duration_in_wins_s'] = self.avg_duration_in_wins_s
        data['avg_duration_in_losses_s'] = self.avg_duration_in_losses_s
        return data


@dataclass
class GameResult:
    game_id: str
    tournament_name: str
    pairing_id: str
    match_num_in_pairing: int
    players_participated: List[Dict[str, Any]]
    winner_id: Optional[str]
    winner_name: Optional[str]
    num_turns: int
    duration_seconds: float
    game_config: Dict[str, Any]
    error_occurred: bool = False
    error_message: Optional[str] = None
    game_state_json_path: Optional[str] = None
    player_end_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Tee(object):
    def __init__(self, *files):
        self.files = [f for f in files if f is not None]

    def write(self, obj):
        for f_obj in self.files:
            if hasattr(f_obj, 'write'):
                f_obj.write(obj)
                if hasattr(f_obj, 'flush'):
                    f_obj.flush()

    def flush(self):
        for f_obj in self.files:
            if hasattr(f_obj, 'flush'):
                f_obj.flush()

@contextlib.contextmanager
def MutedLogger(logger_name: str):
    logger_instance = logging.getLogger(logger_name)
    original_level = logger_instance.getEffectiveLevel()
    logger_instance.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        logger_instance.setLevel(original_level)

@contextlib.contextmanager
def redirect_stdout_tee(filepath: Optional[str], original_stdout_too: bool = True):
    original_stdout = sys.stdout
    file_handle = None
    if filepath:
        file_handle = open(filepath, 'w', encoding='utf-8')

    tee_targets = [f for f in [file_handle, original_stdout if original_stdout_too else None] if f is not None]
    tee = Tee(*tee_targets) if tee_targets else io.StringIO()

    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = original_stdout
        if file_handle:
            file_handle.close()

class GameRunner:
    def __init__(self, game_id: str, tournament_name: str, pairing_id: str, match_num: int,
                 players: List[Player], game_specific_config: Dict[str, Any],
                 output_dir: str, agent_timeout_seconds: float):
        self.game_id = game_id
        self.tournament_name = tournament_name
        self.pairing_id = pairing_id
        self.match_num = match_num
        self.players = players
        self.game_specific_config = game_specific_config
        self.output_dir = output_dir
        self.agent_timeout_seconds = agent_timeout_seconds

        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = self._setup_game_logging()
        self.env = MonopolyEnvironment(
            max_turns=game_specific_config.get('max_turns_per_game', 1000),
        )
        self.game_states_history: List[Dict[str, Any]] = []

    def _setup_game_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"GameRunner.{self.game_id}")
        logger.handlers = []
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(f'%(asctime)s - %(levelname)s - Game {self.game_id} - %(message)s')

        fh_main = logging.FileHandler(os.path.join(self.output_dir, "game_main.log"), encoding='utf-8')
        fh_main.setFormatter(formatter)
        logger.addHandler(fh_main)
        return logger

    def _get_player_details(self, player: Player) -> Dict[str, Any]:
        details = {
            "id": getattr(player, 'mgn_code', 'unknown_id'), # Player objects MUST have mgn_code
            "name": getattr(player, 'name', 'UnknownPlayer'), # Player objects MUST have name
            "type": player.__class__.__name__
        }

        if hasattr(player, 'model_type') and player.model_type:
            details['model_type'] = str(player.model_type)
        if hasattr(player, 'variant') and player.variant:
            details['variant'] = str(player.variant.value)
        return details
        
    def run_game(self) -> GameResult:
        game_stdout_log_path = os.path.join(self.output_dir, "game_stdout.log")
        with redirect_stdout_tee(game_stdout_log_path, original_stdout_too=True):
            self.logger.info(f"Starting game: {self.game_id}")
            player_details_log = [self._get_player_details(p) for p in self.players]
            self.logger.info(f"Players: {json.dumps(player_details_log)}")
            self.logger.info(f"Game Config: {json.dumps(self.game_specific_config)}")

            start_time = time.time()
            current_turn = 0
            error_occurred = False
            error_message = None

            try:
                current_state = self.env.reset(self.players)
                if hasattr(current_state, 'to_dict') and callable(current_state.to_dict):
                    self.game_states_history.append(current_state.to_dict())
                else:
                    self.logger.warning("Initial state object does not have a callable to_dict() method. History will be limited.")

                while not self.env.is_game_over():
                    current_turn += 1
                    current_player_obj = self.env.get_current_player()
                    if not current_player_obj:
                        self.logger.error("Environment returned no current player. Ending game.")
                        error_occurred = True; error_message = "No current player returned by environment."
                        break
                    
                    player_name = getattr(current_player_obj, 'name', 'UnknownPlayer')
                    player_id = getattr(current_player_obj, 'mgn_code', 'unknown_id')
                    self.logger.info(f"Turn {current_turn}: Player {player_name}'s ({player_id}) turn.")

                    action_decision_start_time = time.time()
                    actions: List[Action] = current_player_obj.decide_actions(current_state)
                    action_decision_duration = time.time() - action_decision_start_time

                    if action_decision_duration > self.agent_timeout_seconds:
                        self.logger.warning(f"Player {player_name} exceeded timeout ({action_decision_duration:.2f}s).")

                    if not actions:
                        self.logger.warning(f"Player {player_name} returned no actions.")
                    

                    is_llm_agent = hasattr(current_player_obj, 'is_llm') and callable(getattr(current_player_obj, 'is_llm')) and current_player_obj.is_llm()
                    if is_llm_agent:
                        if hasattr(current_player_obj, 'get_last_prompt_details') and callable(getattr(current_player_obj, 'get_last_prompt_details')):
                            self.logger.info(f"LLM Prompt for {player_name}: {json.dumps(current_player_obj.get_last_prompt_details())}")
                        if hasattr(current_player_obj, 'get_last_llm_response') and callable(getattr(current_player_obj, 'get_last_llm_response')):
                            self.logger.info(f"LLM Response for {player_name}: {json.dumps(current_player_obj.get_last_llm_response())}")
                    self.logger.info(f"Player {player_name} decided action(s) in {action_decision_duration:.2f}s.")

                    for action_idx, action_to_take in enumerate(actions):
                        action_details_str = str(vars(action_to_take)) if action_to_take and hasattr(action_to_take, '__dict__') else str(action_to_take)
                        self.logger.info(f"Turn {current_turn}, Sub-action {action_idx+1}: Player {player_name} takes action: {action_to_take.__class__.__name__} (Details: {action_details_str})")
                        
                        if self.env.is_game_over(): break
                        current_state, _, game_over_after_action, _ = self.env.step(action_to_take)
                        
                        if hasattr(current_state, 'to_dict') and callable(current_state.to_dict):
                            self.game_states_history.append(current_state.to_dict())
                        if game_over_after_action: break
                    
                    if self.env.is_game_over(): break

            except Exception as e:
                self.logger.error(f"Error during game {self.game_id}: {e}", exc_info=True)
                error_occurred = True; error_message = str(e)
            

            winner_obj: Optional[Player] = None
            if hasattr(self.env, 'winner') and self.env.winner is not None:
                winner_obj = self.env.winner
            elif hasattr(self.env.state, 'winner') and self.env.state.winner is not None:
                winner_obj = self.env.state.winner
            elif self.env.is_game_over():
                active_players = []
                if hasattr(self.env.state, 'players') and isinstance(self.env.state.players, list):
                    for p in self.env.state.players:
                        if hasattr(p, 'is_bankrupt'):
                            if not p.is_bankrupt:
                                active_players.append(p)
                        else:
                            active_players.append(p)
                
                if len(active_players) == 1:
                    winner_obj = active_players[0]
                elif not active_players and len(self.env.state.players or []) > 0:
                     self.logger.warning(f"Game {self.game_id} ended with all players bankrupt or removed.")
            
            winner_id = getattr(winner_obj, 'mgn_code', None) if winner_obj else None
            winner_name = getattr(winner_obj, 'name', 'None/Draw') if winner_obj else 'None/Draw'

            self.logger.info(f"Game {self.game_id} ended. Winner: {winner_name}. Turns: {current_turn}.")
            duration_seconds = time.time() - start_time
            self.logger.info(f"Game duration: {duration_seconds:.2f} seconds.")

            game_state_json_path = os.path.join(self.output_dir, "game_full_state.json")
            game_json_data = {
                "metadata": {
                    "game_id": self.game_id, "tournament_name": self.tournament_name,
                    "pairing_id": self.pairing_id, "match_num": self.match_num,
                    "players": player_details_log, "game_config": self.game_specific_config,
                    "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.datetime.now().isoformat(),
                    "duration_seconds": duration_seconds, "num_turns": current_turn,
                    "winner_id": winner_id, "winner_name": winner_name,
                    "error_occurred": error_occurred, "error_message": error_message
                },
                "game_log": self.game_states_history
            }
            with open(game_state_json_path, 'w', encoding='utf-8') as f_json:
                json.dump(game_json_data, f_json, indent=2)
            self.logger.info(f"Full game state and log saved to {game_state_json_path}")

            player_end_states = {}
            current_game_state_obj = self.env.state
            if hasattr(current_game_state_obj, 'to_dict') and callable(current_game_state_obj.to_dict):
                final_env_state_dict = current_game_state_obj.to_dict()
                if 'players' in final_env_state_dict and isinstance(final_env_state_dict['players'], list):
                    for p_state_data in final_env_state_dict['players']:
                        if isinstance(p_state_data, dict):
                            p_id = p_state_data.get('mgn_code', p_state_data.get('id'))
                            if p_id:
                                player_end_states[p_id] = {k: v for k, v in p_state_data.items() if k not in ['name', 'id', 'mgn_code', 'type']}
                        elif hasattr(p_state_data, 'mgn_code'): # If p_state_data are Player objects themselves
                            p_id = p_state_data.mgn_code
                            player_end_states[p_id] = {
                                attr: getattr(p_state_data, attr) for attr in dir(p_state_data)
                                if not callable(getattr(p_state_data, attr)) and not attr.startswith('_')
                                and attr not in ['name', 'id', 'mgn_code', 'type', 'action_manager'] # Exclude methods, private, basic ID, and complex objects
                            }

            return GameResult(
                game_id=self.game_id, tournament_name=self.tournament_name,
                pairing_id=self.pairing_id, match_num_in_pairing=self.match_num,
                players_participated=player_details_log, winner_id=winner_id, winner_name=winner_name,
                num_turns=current_turn, duration_seconds=duration_seconds,
                game_config=self.game_specific_config, error_occurred=error_occurred,
                error_message=error_message, game_state_json_path=game_state_json_path,
                player_end_states=player_end_states
            )


class Tournament:
    def __init__(self, agents: List[Player], config: TournamentConfig, output_folder_base: str = "tournament_results"):
        self.agents = agents
        self.config = config
        self.tournament_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tournament_folder_name = f"tournament_{self.config.name.replace(' ', '_')}_{self.tournament_id[:8]}_{timestamp}"
        self.output_dir = os.path.join(output_folder_base, self.tournament_folder_name)
        self.games_output_dir = os.path.join(self.output_dir, "games")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.games_output_dir, exist_ok=True)

        self.logger = self._setup_tournament_logging()
        self.agent_stats: Dict[str, AgentStats] = {
            getattr(p, 'mgn_code', f"unknown_id_{idx}"): AgentStats(
                agent_id=getattr(p, 'mgn_code', f"unknown_id_{idx}"),
                agent_name=getattr(p, 'name', f"UnknownAgent_{idx}"),
                agent_type=p.__class__.__name__
            ) for idx, p in enumerate(self.agents)
        }
        self.game_results: List[GameResult] = []

    def _setup_tournament_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"Tournament.{self.config.name}.{self.tournament_id[:8]}")
        logger.handlers = []
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh_main = logging.FileHandler(os.path.join(self.output_dir, "tournament_main.log"), encoding='utf-8')
        fh_main.setFormatter(formatter)
        logger.addHandler(fh_main)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def _generate_pairings(self) -> List[Tuple[Player, ...]]:
        pairings: List[Tuple[Player, ...]] = []
        min_p, max_p = self.config.num_players_per_game_range
        
        if not self.agents:
            self.logger.warning("No agents provided for the tournament.")
            return []

        for k_players in range(min_p, min(max_p + 1, len(self.agents) + 1)):
            if self.config.pairing_strategy == "all_vs_all":
                for combo in itertools.combinations(self.agents, k_players):
                    pairings.append(combo)
            elif self.config.pairing_strategy == "random_groups":
                for _ in range(self.config.num_random_games_if_strategy_random):
                    pairings.append(tuple(random.sample(self.agents, k_players)))
            else:
                self.logger.error(f"Unknown pairing strategy: {self.config.pairing_strategy}")
                raise ValueError(f"Unknown pairing strategy: {self.config.pairing_strategy}")
        
        if not pairings:
            self.logger.warning("No valid pairings generated. Check agent count and player range.")
        return pairings

    def _save_tournament_config(self):
        config_path = os.path.join(self.output_dir, "tournament_config.json")

        agent_configs_serializable = [GameRunner( # Temporary instance to call static-like method
            game_id='', tournament_name='', pairing_id='', match_num=0, players=[],
            game_specific_config={}, output_dir=self.output_dir, agent_timeout_seconds=0
        )._get_player_details(agent) for agent in self.agents]


        full_config_to_save = {
            "tournament_settings": self.config.to_dict(),
            "agents_configured": agent_configs_serializable,
            "tournament_id": self.tournament_id,
            "output_directory": self.output_dir
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(full_config_to_save, f, indent=2)
        self.logger.info(f"Tournament configuration saved to {config_path}")

    def _update_agent_stats(self, result: GameResult):
        for p_detail in result.players_participated:
            agent_id = p_detail["id"]
            if agent_id in self.agent_stats:
                stats = self.agent_stats[agent_id]
                stats.games_played += 1
                if result.winner_id == agent_id:
                    stats.wins += 1
                    stats.total_turns_in_wins += result.num_turns
                    stats.total_duration_in_wins_s += result.duration_seconds
                else:
                    stats.total_turns_in_losses += result.num_turns
                    stats.total_duration_in_losses_s += result.duration_seconds
            else:
                self.logger.warning(f"Agent ID {agent_id} from game result not found in initial agent_stats. This is unexpected.")
    
    def _save_results(self):
        summary_path = os.path.join(self.output_dir, "tournament_summary.csv")
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            if not self.game_results:
                 f.write("No game results to summarize.\n")
            else:
                fieldnames = list(self.game_results[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.game_results:
                    writer.writerow(result.to_dict())
        self.logger.info(f"Tournament summary saved to {summary_path}")

        agent_stats_path = os.path.join(self.output_dir, "agent_statistics.csv")
        with open(agent_stats_path, 'w', newline='', encoding='utf-8') as f:
            if not self.agent_stats:
                f.write("No agent statistics to save.\n")
            else:
                fieldnames = list(next(iter(self.agent_stats.values())).to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for stats in self.agent_stats.values():
                    writer.writerow(stats.to_dict())
        self.logger.info(f"Agent statistics saved to {agent_stats_path}")

    def run(self):
        self.logger.info(f"Starting Tournament: {self.config.name} (ID: {self.tournament_id})")
        self.logger.info(f"Output Directory: {self.output_dir}")
        self._save_tournament_config()

        pairings = self._generate_pairings()
        if not pairings:
            self.logger.warning("No game pairings generated. Tournament will not run any games.")
            self._save_results(); return

        self.logger.info(f"Generated {len(pairings)} unique player groupings. Each will run {self.config.num_matches_per_pairing} time(s).")
        total_games_to_run = len(pairings) * self.config.num_matches_per_pairing
        self.logger.info(f"Total games to run: {total_games_to_run}")

        game_counter = 0
        for i, player_group_tuple in enumerate(pairings):
            player_names_for_id = sorted([getattr(p, 'name', f'A{idx}').replace(' ','_')[:10] for idx,p in enumerate(player_group_tuple)])
            pairing_id_suffix = '_vs_'.join(player_names_for_id)
            pairing_id = f"pairing_{i+1:03d}_{pairing_id_suffix}"
            
            self.logger.info(f"Running matches for pairing {i+1}/{len(pairings)} ({pairing_id}): {[getattr(p,'name','?') for p in player_group_tuple]}")

            for match_num in range(self.config.num_matches_per_pairing):
                game_counter += 1
                game_id_suffix = f"{pairing_id}_match{match_num+1:02d}"
                game_id = f"{self.tournament_id[:8]}_{game_id_suffix}"
                game_output_dir = os.path.join(self.games_output_dir, f"game_{game_id_suffix}")
                
                current_players_for_game = list(player_group_tuple)
                if self.config.shuffle_turn_order_in_redundant_matches and len(current_players_for_game) > 1:
                    random.shuffle(current_players_for_game)
                    self.logger.info(f"Shuffled player order for game {game_id}: {[getattr(p,'name','?') for p in current_players_for_game]}")

                self.logger.info(f"Starting Game {game_counter}/{total_games_to_run} (ID: {game_id})")
                
                game_specific_config = {"max_turns_per_game": self.config.max_turns_per_game}
                runner = GameRunner(
                    game_id=game_id, tournament_name=self.config.name, pairing_id=pairing_id,
                    match_num=match_num + 1, players=current_players_for_game,
                    game_specific_config=game_specific_config, output_dir=game_output_dir,
                    agent_timeout_seconds=self.config.agent_timeout_seconds
                )
                game_result = runner.run_game()
                self.game_results.append(game_result)
                self._update_agent_stats(game_result)
                self.logger.info(f"Finished Game {game_id}. Winner: {game_result.winner_name}. Turns: {game_result.num_turns}.")
                
                if game_counter > 0 and game_counter % 10 == 0 and game_counter < total_games_to_run:
                     self.logger.info(f"Saving intermediate results after {game_counter} games...")
                     self._save_results()

        self.logger.info("Tournament finished.")
        self._save_results()

if __name__ == "__main__":
    print("Setting up example tournament...")
    try:
        from monopoly_gym.players.random import RandomPlayer
        from monopoly_gym.players.random_masked import MaskedRandomPlayer
    except ImportError as e:
        print(f"Warning: Could not import all desired agent classes: {e}. Using basic Player placeholders for demo if necessary.")
        class RandomPlayer(Player):
            def __init__(self, name: str, mgn_code: Optional[str] = None, **kwargs):
                self.name = name
                self.mgn_code = mgn_code or name.lower().replace(" ", "_")
                super().__init__(name=self.name, mgn_code=self.mgn_code, **kwargs)
            def decide_actions(self, state: State) -> List[Action]: return [] 
        class MaskedRandomPlayer(Player):
             def __init__(self, name: str, mgn_code: Optional[str] = None, **kwargs):
                self.name = name
                self.mgn_code = mgn_code or name.lower().replace(" ", "_")
                super().__init__(name=self.name, mgn_code=self.mgn_code, **kwargs)
             def decide_actions(self, state: State) -> List[Action]: return []

    agents_list = [
        RandomPlayer(name="Randy", mgn_code="randy001"),
        MaskedRandomPlayer(name="Masky", mgn_code="masky002"),
        RandomPlayer(name="Loopy", mgn_code="loopy003"),
    ]

    if len(agents_list) < 2:
        print("Error: At least 2 agents are required for a tournament. Please add more agents to agents_list."); sys.exit(1)

    config1 = TournamentConfig(
        name="ICML_TestRun_2P", num_players_per_game_range=(2, 2), 
        num_matches_per_pairing=2, pairing_strategy="all_vs_all",
        max_turns_per_game=150, agent_timeout_seconds=30.0
    )
    tournament1 = Tournament(agents=agents_list, config=config1, output_folder_base="my_tournament_output")
    print(f"Tournament 1 configured. Output dir: {tournament1.output_dir}"); tournament1.run()
    print(f"Tournament 1 finished. Results in: {tournament1.output_dir}")

    if len(agents_list) >= 3:
        print("\nConfiguring tournament with 3 players...")
        config2 = TournamentConfig(
            name="ICML_TestRun_3P", num_players_per_game_range=(3, 3),
            num_matches_per_pairing=1, pairing_strategy="all_vs_all",
            max_turns_per_game=200,
        )
        tournament2 = Tournament(agents=agents_list, config=config2, output_folder_base="my_tournament_output")
        print(f"Tournament 2 configured. Output dir: {tournament2.output_dir}"); tournament2.run()
        print(f"Tournament 2 finished. Results in: {tournament2.output_dir}")
    else:
        print("\nSkipping 3-player tournament example as there are fewer than 3 agents.")