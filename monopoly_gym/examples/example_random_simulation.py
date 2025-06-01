# monopoly_gym/gym/examples/example_random_simulation.py
from gym.env import MonopolyEnvironment
from gym.players.random_masked import MaskedRandomPlayer, ActionSpaceType

def main():
    env = MonopolyEnvironment(use_render=True)
    env.reset()
    
    agents = [
        MaskedRandomPlayer(name="AI 1", mgn_code="A1", action_space_type=ActionSpaceType.HIERARCHICAL),
        MaskedRandomPlayer(name="AI 2", mgn_code="A2", action_space_type=ActionSpaceType.HIERARCHICAL),
        MaskedRandomPlayer(name="AI 3", mgn_code="A3", action_space_type=ActionSpaceType.HIERARCHICAL),
        MaskedRandomPlayer(name="AI 4", mgn_code="A4", action_space_type=ActionSpaceType.HIERARCHICAL)
    ]
    
    env.add_players(agents)
    env.play()

if __name__ == "__main__":
    main()