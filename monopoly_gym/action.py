# monopoly_gym/gym/action.py
from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Union, List, Tuple, Type, TYPE_CHECKING
import numpy as np

from gym.spaces import (
    Box,
    Discrete,
    Dict as GymDict,
    Space,
    MultiBinary,
    Tuple as GymTuple,
)

from monopoly_gym.state import HOTEL_AUCTION_THRESHOLD, HOUSE_AUCTION_THRESHOLD, MAX_HOTELS_AVAILABLE_FOR_AUCTION, MAX_HOUSES_AVAILABLE_FOR_AUCTION, AuctionBid, AuctionState, BuildingType, State, TradeOffer
from monopoly_gym.tile import Property, Street
import logging
import random

if TYPE_CHECKING:
    from monopoly_gym.player import Player
    from monopoly_gym.state import State, TradeOffer, AuctionBid, AuctionState


logger = logging.getLogger(__name__)

MAX_PLAYERS = 8
MAX_CASH = 20580
MAX_PROPERTIES = 28
MAX_STREETS = 22
MAX_BUILD_COUNT = 5
JAIL_BAIL_AMOUNT = 50
VOLUNTARY_BANKRUPTCY = True
MAX_MESSAGE_LENGTH = 500

class ActionSpaceType(Enum):
    FLAT = "flat"
    HIERARCHICAL = "hierarchical"

class Action(ABC):
    def __init__(self, name: str, player: Optional[Player] = None) -> None:
        self.name = name
        self.player = player

    @abstractmethod
    def to_mgn(self) -> str:
        ...

    @abstractmethod
    def process(self, state: State) -> None:
        ...

    @classmethod
    @abstractmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        ...

    @classmethod
    @abstractmethod
    def flat_parameter_size(cls) -> int:
        ...

    @classmethod
    @abstractmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        ...

    @classmethod
    @abstractmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        ...

    @abstractmethod
    def to_dict(self) -> Dict:
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict, state: State) -> Action:
        ...

class ProposeTradeAction(Action):
    """
    Player proposes a trade to a specific responder,
    offering 'give' and asking for 'receive'.
    """

    def __init__(self, trade_offer: TradeOffer ):
        super().__init__("ProposeTrade", trade_offer.proposer)
        self.responder = trade_offer.responder
        # offer
        self.cash_offered = trade_offer.cash_offered
        self.properties_offered = trade_offer.properties_offered
        self.get_out_of_jail_cards_offered = trade_offer.get_out_of_jail_cards_offered
        # asking
        self.cash_asking = trade_offer.cash_asking
        self.properties_asking = trade_offer.properties_asking
        self.get_out_of_jail_cards_asking = trade_offer.get_out_of_jail_cards_asking

    def to_mgn(self) -> str:
        """String representation for logging/analysis."""
        properties_offered = self.properties_offered
        cash_offered = self.cash_offered
        properties_asking = self.properties_asking
        cash_asking = self.cash_asking

        give_str = [f"@{g}" for g in properties_offered]
        if cash_offered > 0:
            give_str.append(f"${cash_offered}")
        receive_str = [f"@{r}" for r in properties_asking]
        if cash_asking > 0:
            receive_str.append(f"${cash_asking}")

        give_part = "+".join(give_str) if give_str else "0"
        receive_part = "+".join(receive_str) if receive_str else "0"
        return f"T({self.player.mgn_code}>{self.responder.mgn_code}:{give_part};{receive_part})"

    def process(self, state: State) -> None:
        if state.pending_trade:
            logger.warning("Cannot propose a new trade while another trade is pending.")
            return

        if self.responder not in state.players:
            logger.warning(f"Responder {self.responder.name} not in the game. Trade not proposed.")
            return

        logger.info(f"{self.player.name} proposes a trade to {self.responder.name}.")
        state.pending_trade = TradeOffer(
            proposer=self.player,
            responder=self.responder,
            cash_offered=self.cash_offered,
            cash_asking=self.cash_asking,
            properties_asking=self.properties_asking,
            properties_offered=self.properties_offered,
            get_out_of_jail_cards_asking=self.get_out_of_jail_cards_asking,
            get_out_of_jail_cards_offered=self.get_out_of_jail_cards_offered
        )

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        if state.pending_trade or state.auction_state:
            return [False]
        return [len(state.players) > 1 and not state.pending_trade]

    @classmethod
    def flat_parameter_size(cls) -> int:
        # Could be large if encoding 'give'/'receive' in flatten, but for demonstration:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        num_other_players = len(state.players) - 1

        if state.pending_trade or state.auction_state or num_other_players <= 1:
            partner_mask = [False] * (MAX_PLAYERS - 1)
            return {
                "trade_partner": partner_mask,
                "cash_offered": [False]*(MAX_CASH+1),
                "properties_offered": [False]*MAX_PROPERTIES,
                "get_out_of_jail_cards_offered": [False] * 2,
                "cash_asking": [False]*(MAX_CASH+1),
                "properties_asking": [False]*MAX_PROPERTIES,
                "get_out_of_jail_cards_asking": [False] * 2
            }

        partner_mask = [True] * num_other_players
        partner_mask.extend([False] * (MAX_PLAYERS - 1 - num_other_players))

        give_prop_mask = [False]*MAX_PROPERTIES
        for prop in current_player.properties:
            give_prop_mask[prop.property_idx] = True

        get_out_of_jail_cards_offered_mask = [False] * 2 # Max offer 1 card
        if current_player.jail_free_cards > 0:
            get_out_of_jail_cards_offered_mask[1] = True 
        get_out_of_jail_cards_offered_mask[0] = True 

        receive_prop_mask = [False]*MAX_PROPERTIES
        responder_balances = [p.balance for p in state.players if p != current_player]
        max_cash_asking = max(responder_balances) if responder_balances else 0
        cash_asking_mask = [c <= max_cash_asking for c in range(MAX_CASH + 1)]
        get_out_of_jail_cards_asking_mask = [False] * 2
        for other in state.players:
            if other != current_player:
                for prop in other.properties:
                    receive_prop_mask[prop.property_idx] = True
                if other.jail_free_cards > 0:
                    get_out_of_jail_cards_asking_mask[1] = True
        get_out_of_jail_cards_asking_mask[0] = True

        cash_offered_mask = [c <= current_player.balance for c in range(MAX_CASH+1)]

        return {
            "trade_partner": partner_mask,
            "cash_offered": cash_offered_mask,
            "properties_offered": give_prop_mask,
            "get_out_of_jail_cards_offered": get_out_of_jail_cards_offered_mask,
            "cash_asking": cash_asking_mask,
            "properties_asking": receive_prop_mask,
            "get_out_of_jail_cards_asking": get_out_of_jail_cards_asking_mask
        }


    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {
            "trade_partner": Discrete(MAX_PLAYERS - 1),
            "cash_offered": Discrete(MAX_CASH + 1),
            "get_out_of_jail_cards_offered": Discrete(2),
            "properties_offered": MultiBinary(MAX_PROPERTIES),
            "cash_asking": Discrete(MAX_CASH + 1),
            "properties_asking": MultiBinary(MAX_PROPERTIES),
            "get_out_of_jail_cards_asking": Discrete(2)
        }

    def to_dict(self) -> Dict:
        return {
            "type": "ProposeTrade",
            "proposer_id": self.player.mgn_code,
            "responder_id": self.responder.mgn_code,
            "cash_offered": self.cash_offered,
            "properties_offered_indices": [p.index for p in self.properties_offered],
            "properties_offered_details": [{"name": p.name, "index": p.index} for p in self.properties_offered],
            "get_out_of_jail_cards_offered": self.get_out_of_jail_cards_offered,
            "cash_asking": self.cash_asking,
            "properties_asking_indices": [p.index for p in self.properties_asking],
            "properties_asking_details": [{"name": p.name, "index": p.index} for p in self.properties_asking],
            "get_out_of_jail_cards_asking": self.get_out_of_jail_cards_asking,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> ProposeTradeAction:
        proposer_mgn_code = data.get("proposer_id", data.get("mgn_code"))
        proposer = next(p for p in state.players if p.mgn_code == proposer_mgn_code)
        responder = next(p for p in state.players if p.mgn_code == data["responder_id"])

        properties_offered = []
        for idx in data.get("properties_offered_indices", []):
            if 0 <= idx < len(state.board.board) and isinstance(state.board.board[idx], Property):
                properties_offered.append(state.board.board[idx])
            else:
                logger.warning(f"Invalid property index {idx} in properties_offered_indices for ProposeTradeAction.")

        properties_asking = []
        for idx in data.get("properties_asking_indices", []):
            if 0 <= idx < len(state.board.board) and isinstance(state.board.board[idx], Property):
                properties_asking.append(state.board.board[idx])
            else:
                logger.warning(f"Invalid property index {idx} in properties_asking_indices for ProposeTradeAction.")

        return ProposeTradeAction(
            trade_offer=TradeOffer(
                proposer=proposer,
                responder=responder,
                cash_offered=int(data.get("cash_offered", 0)),
                properties_offered=properties_offered,
                get_out_of_jail_cards_offered=int(data.get("get_out_of_jail_cards_offered", 0)),
                cash_asking=int(data.get("cash_asking", 0)),
                properties_asking=properties_asking,
                get_out_of_jail_cards_asking=int(data.get("get_out_of_jail_cards_asking", 0)),
            )
        )
class RollDiceAction(Action):
    """Action for rolling dice and moving the player."""
    
    def __init__(self, player: Player):
        super().__init__("RollDice", player)
        self.dice_roll: Tuple[int, int] = (0, 0)
        self.rolled_doubles: bool = False

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} ROLL"

    def process(self, state: State) -> None:
        logger.info(f"[ACTION] {self.player.name} attempts to ROLL dice.")
        state.rolled_this_turn = True
        d1, d2 = self.dice_roll
        if self.dice_roll == (0, 0):
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            self.dice_roll = (d1, d2)
            logger.info(f"[ACTION] {self.player.name} rolled dice: {self.dice_roll}")
        
        self.rolled_doubles = (d1 == d2)
        
        if self.rolled_doubles:
            state.current_consecutive_doubles += 1
        else:
            state.current_consecutive_doubles = 0
        
        if state.current_consecutive_doubles >= 3:
            logger.info(f"[ACTION] {self.player.name} rolled three consecutive doubles! Going to jail.")
            state.send_player_to_jail(self.player)
            state.rolled_this_turn = True  
            return
        
        old_pos = self.player.position
        total = d1 + d2
        self.player.position = (old_pos + total) % len(state.board.board)
        
        if old_pos + total >= len(state.board.board):
            self.player.balance += 200
            logger.info(f"[ACTION] {self.player.name} passed Go, +$200 => {self.player.balance}")
        
        logger.info(f"[ACTION] {self.player.name} moves from {old_pos} to {self.player.position}")
        state.handle_landing_on_tile(player=self.player, dice_roll=(d1, d2))

        if self.rolled_doubles:
            logger.info(f"[ACTION] {self.player.name} rolled doubles; may roll again.")
        else:
            state.rolled_this_turn = True
            logger.info(f"[ACTION] {self.player.name} normal roll => done rolling for this turn.")


    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()

        if current_player.in_jail or state.auction_state is not None or state.pending_trade is not None:
            valid = False
        else:
            valid = (not state.rolled_this_turn)
        return [valid]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 0 

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        if state == None:
            return
        current_player = state.current_player()
        if current_player.in_jail or state.auction_state is not None or state.pending_trade is not None:
            valid = False
        else:
            valid = (not state.rolled_this_turn)

        return {"roll": [valid]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {} 

    def to_dict(self) -> Dict:
        return {
            "type": "RollDice",
            "mgn_code": self.player.mgn_code,
            "dice_roll": self.dice_roll,
            "rolled_doubles": self.rolled_doubles,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> RollDiceAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        action = RollDiceAction(player)
        action.dice_roll = tuple(data.get("dice_roll", (0, 0)))
        action.rolled_doubles = data.get("rolled_doubles", False)
        return action


class EndTurnAction(Action):
    def __init__(self, player: Player):
        super().__init__("EndTurn", player)

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} ENDTURN"

    def process(self, state: State) -> None:
        state.rolled_this_turn = False
        if state.current_consecutive_doubles == 0:
            if state.auction_state:
                state.advance_auction_turn(self.player)
            else:
                print("Advancing turn")
                state.advance_turn(self.player)

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        tile = state.board.board[current_player.position]
        must_decide_property = (
            isinstance(tile, Property) 
            and tile.owner is None 
            and state.auction_state is None
            and state.rolled_this_turn is True
        )
        has_negative_balance = current_player.balance < 0
        in_auction = not state.auction_state is None
        valid = not must_decide_property and not in_auction and not has_negative_balance and state.pending_trade is None and state.rolled_this_turn
        return [valid]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        tile = state.board.board[current_player.position]
        must_decide_property = (
            isinstance(tile, Property) 
            and tile.owner is None 
            and state.auction_state is None
        )
        has_negative_balance = current_player.balance < 0
        in_auction = not state.auction_state is None
        valid = not must_decide_property and not in_auction and not has_negative_balance and state.pending_trade is None and state.rolled_this_turn == True
        return {"valid": [valid]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"valid": Discrete(1)}

    def to_dict(self) -> Dict:
        return {"type": "EndTurn", "mgn_code": self.player.mgn_code}

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> EndTurnAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        return EndTurnAction(player)

class BuyAction(Action):
    def __init__(self, player: Player, property: Property):
        super().__init__("Buy", player)
        self.property = property
        self.price = property.purchase_cost

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} B@{self.property.index}:${self.price}"

    def process(self, state: State) -> None:
        if self.player.balance >= self.price and self.property.owner is None:
            self.property.owner = self.player
            self.player.balance -= self.price
            self.player.properties.append(self.property)

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        tile = state.board.board[current_player.position]
        valid = (
            isinstance(tile, Property)
            and tile.owner is None
            and current_player.balance >= tile.purchase_cost
            and state.auction_state is None
            and state.pending_trade is None
            and not state.property_decision_made_this_landing
        )
        return [valid]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        tile = state.board.board[current_player.position]
        valid = (
            isinstance(tile, Property)
            and tile.owner is None
            and current_player.balance >= tile.purchase_cost
            and state.auction_state is None
            and state.pending_trade is None
            and not state.property_decision_made_this_landing
        )
        return {"property": [valid]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"property": Discrete(1)}

    def to_dict(self) -> Dict:
        return {
            "type": "Buy",
            "mgn_code": self.player.mgn_code,
            "property_index": self.property.index,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> BuyAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        prop = state.board.board[data["property_index"]]
        return BuyAction(player, prop)

class AuctionAction(Action):
    def __init__(self, player: Player, property: Property):
        super().__init__("Auction", player)
        self.property = property

    def to_mgn(self) -> str:
        return f"AU@{self.property.index}"

    def process(self, state: State) -> None:
        state.auction_state = AuctionState(
            aucition_item=self.property,
            participants=[player for player in state.players],
            bids=[],
            initial_bidding_index=state.current_player_index
        )

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        tile = state.board.board[current_player.position]
        valid = isinstance(tile, Property) and tile.owner is None  and state.auction_state is None and state.pending_trade is None and not state.property_decision_made_this_landing
        return [valid]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        tile = state.board.board[current_player.position]
        valid = isinstance(tile, Property) and tile.owner is None and state.auction_state is None and state.pending_trade is None and not state.property_decision_made_this_landing
        return {"auction_item": [valid]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"auction_item": Discrete(1)}

    def to_dict(self) -> Dict:
        return {
            "type": "Auction",
            "mgn_code": self.player.mgn_code,
            "property_index": self.property.index,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> AuctionAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        prop = state.board.board[data["property_index"]]
        return AuctionAction(player, prop)

class AuctionBidAction(Action):
    def __init__(self, player: Player, bid_amount: int):
        super().__init__("AuctionBid", player)
        self.bid_amount = bid_amount

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code}.${self.bid_amount}"

    def process(self, state: State) -> None:
        #print("Processing auction bid action")
        if state.auction_state:
            state.auction_state.bids.append(AuctionBid(self.player, self.bid_amount))
        if state.auction_state.is_done():
            #print("Auctioning is done. resolving.")
            state.auction_state.resolve(state)
        else:
            #print(f"Auctioning is not done. proceeding. auction_state={state.auction_state.to_dict()}")
            state.advance_auction_turn(player=self.player)
            #print(f"Advanced turn. new auction_state={state.auction_state.to_dict()}")


    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        if not state.auction_state:
            return [False] * (MAX_CASH + 1)

        current_player = state.auction_state.participants[state.auction_state.current_bidder_index]
        highest_bid = state.auction_state.highest_bid()
        min_bid = (highest_bid.bid_amount + 1) if highest_bid else 1
        max_bid = current_player.balance

        return [(min_bid <= i <= max_bid) for i in range(MAX_CASH + 1)]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return MAX_CASH + 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        if not state.auction_state:
            return {"bid_amount": [False]*(MAX_CASH+1)}

        current_player = state.auction_state.participants[state.auction_state.current_bidder_index]
        highest_bid = state.auction_state.highest_bid()
        min_bid = (highest_bid.bid_amount + 1) if highest_bid else 1
        max_bid = current_player.balance

        return {
            "bid_amount": [(min_bid <= i <= max_bid) for i in range(MAX_CASH + 1)]
        }

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"bid_amount": Discrete(MAX_CASH + 1)}

    def to_dict(self) -> Dict:
        return {
            "type": "AuctionBid",
            "mgn_code": self.player.mgn_code,
            "bid_amount": self.bid_amount,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> AuctionBidAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        return AuctionBidAction(player, data["bid_amount"])

class MortgageAction(Action):
    def __init__(self, player: Player, property: Property):
        super().__init__("Mortgage", player)
        self.property = property

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} MG@{self.property.index}:${self.property.mortgage_price}"

    def process(self, state: State) -> None:
        #print(f"attempting to mortgage {self.property.to_dict()}")
        if not self.property.is_mortgaged:
            self.property.is_mortgaged = True
            self.player.balance += self.property.mortgage_price

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        mask = [False] * MAX_PROPERTIES
        if state.auction_state is not None or state.pending_trade is not None:
            return mask
        for prop in current_player.properties:
            if isinstance(prop, Property) and not prop.is_mortgaged:
                mask[prop.property_idx] = True
        return mask


    @classmethod
    def flat_parameter_size(cls) -> int:
        return MAX_PROPERTIES

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        prop_mask = [False] * MAX_PROPERTIES
        if state.auction_state is not None or state.pending_trade is not None:
            return {"property": prop_mask}
        for prop in current_player.properties:
            if isinstance(prop, Property) and not prop.is_mortgaged:
                prop_mask[prop.property_idx] = True
        return {"property": prop_mask}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"property": Discrete(MAX_PROPERTIES)}

    def to_dict(self) -> Dict:
        return {
            "type": "Mortgage",
            "mgn_code": self.player.mgn_code,
            "property_index": self.property.index,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> MortgageAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        prop = state.board.board[data["property_index"]]
        return MortgageAction(player, prop)

class UnmortgageAction(Action):
    def __init__(self, player: Player, property: Property):
        super().__init__("Unmortgage", player)
        self.property = property

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} UM@{self.property.index}:${self.property.unmortgage_price}"

    def process(self, state: State) -> None:
        #print(f"current player is = {state.current_player().mgn_code}")
        #print(f"attempting to unmortgage {self.property.to_dict()}")
        if self.property.is_mortgaged and self.player.balance >= self.property.unmortgage_price:
            self.property.is_mortgaged = False
            self.player.balance -= self.property.unmortgage_price

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        mask = [False] * MAX_PROPERTIES
        if state.auction_state is not None or state.pending_trade is not None:
            return mask
        for prop in current_player.properties:
            if isinstance(prop, Property) and prop.is_mortgaged and current_player.balance >= prop.unmortgage_price:
                mask[prop.property_idx] = True
        return mask

    @classmethod
    def flat_parameter_size(cls) -> int:
        return MAX_PROPERTIES

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        prop_mask = [False] * MAX_PROPERTIES
        if state.auction_state is not None or state.pending_trade is not None:
            return {"property": prop_mask}
        for prop in current_player.properties:
            if isinstance(prop, Property) and prop.is_mortgaged and current_player.balance >= prop.unmortgage_price:
                prop_mask[prop.property_idx] = True
        return {"property": prop_mask}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"property": Discrete(MAX_PROPERTIES)}

    def to_dict(self) -> Dict:
        return {
            "type": "Unmortgage",
            "mgn_code": self.player.mgn_code,
            "property_index": self.property.index,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> UnmortgageAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        prop = state.board.board[data["property_index"]]
        return UnmortgageAction(player, prop)

class BuildAction(Action):
    def __init__(self, player: Player, street: Street, quantity: int):
        super().__init__("Build", player)
        self.street = street
        self.quantity = quantity

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} H@{self.street.index}x{self.quantity}"

    def process(self, state: State) -> None:
        current_player = self.player

        if not isinstance(self.street, Street):
            raise Exception(f"Cannot build on {self.street.name}, it is not a Street.")

        if self.street.owner != self.player:
            raise Exception(f"{self.player.name} does not own {self.street.name}.")

        if self.street.is_mortgaged:
            raise Exception(f"Cannot build on {self.street.name}, it is mortgaged.")

        if self.street.hotels >= 1:
            raise Exception(f"Already has a hotel on {self.street.name}")



        if state.auction_state and \
        state.auction_state.placing_building_after_win and \
        len(state.auction_state.bids) > 0 and state.auction_state.highest_bid().bidder == current_player:

            if self.quantity != 1:
                logger.error(f"Invalid BuildAction: {current_player.name} must place exactly 1 building won from auction, tried {self.quantity}.")
                return
            
            building_to_place = BuildingType.HOUSE if state.auction_state.auction_item == BuildingType.HOUSE else BuildingType.HOTEL
            
            if not state.player_can_build_on_property(current_player, self.street, building_to_place, check_even_build=True):
                logger.error(f"Invalid BuildAction: {current_player.name} cannot place auctioned {building_to_place.name} on {self.street.name} (idx {self.street.index}). Check ownership, mortgage, set completion, space, or even-build.")
                return

            logger.info(f"{current_player.name} is placing the {building_to_place.name} won in auction on {self.street.name}.")
            
            if building_to_place == BuildingType.HOUSE:
                self.street.houses += 1
            elif building_to_place == BuildingType.HOTEL:
                if self.street.houses == 4:
                    state.houses_available += 4
                    self.street.houses = 0
                    self.street.hotels += 1
                else:
                    logger.error(f"CRITICAL: {current_player.name} trying to place hotel on {self.street.name} which doesn't have 4 houses.")
                    state.auction_state = None
                    return
                
            state.auction_state = None
            return

        if not isinstance(self.street, Street) or self.street.owner != current_player:
            logger.warning(f"{current_player.name} cannot build on {self.street.name}: not a street or not owned.")
            return
        if self.street.is_mortgaged:
            logger.warning(f"{self.player.name} cannot build on mortgaged {self.street.name}.")
            return
        if not state.player_has_complete_color_set(current_player, self.street.color_set):
            logger.warning(f"{current_player.name} cannot build on {self.street.name}: does not own full color set.")
            return
        building_type_attempted: Optional[BuildingType] = None
        cost_of_one_building: int = 0
        current_houses_on_street = self.street.houses
        
        if current_houses_on_street < 4 and (current_houses_on_street + self.quantity <= 4):
            building_type_attempted = BuildingType.HOUSE
            cost_of_one_building = self.street.color_set.house_cost
        elif current_houses_on_street == 4 and self.quantity >= 1 and self.street.hotels == 0: # Trying to build hotel
            building_type_attempted = BuildingType.HOTEL
            cost_of_one_building = self.street.color_set.hotel_cost
        elif current_houses_on_street < 4 and (current_houses_on_street + self.quantity > 4) and self.street.hotels == 0:
            building_type_attempted = BuildingType.HOUSE
            cost_of_one_building = self.street.color_set.house_cost
            logger.info(f"Build attempt for {self.quantity} on {self.street.name} (currently {current_houses_on_street} houses) will be treated as a {building_type_attempted.name} auction if shortage.")
        else:
            logger.warning(f"{current_player.name} invalid build attempt on {self.street.name} with {self.quantity} units (current: H{self.street.houses} HTL{self.street.hotels}).")
            return

        if building_type_attempted == BuildingType.HOUSE and self.quantity > 1:
            new_total = self.street.houses + self.quantity
            same_color_streets = [
                s for s in state.board.board
                if isinstance(s, Street) and s.color_set == self.street.color_set
            ]
            for other in same_color_streets:
                if other is not self.street and other.houses < new_total - 1:
                    raise Exception(f"Cannot build due to even-build rule on {other.name}")

        else:
            if not state.player_can_build_on_property(current_player,
                                                    self.street,
                                                    building_type_attempted,
                                                    check_even_build=True):
                raise Exception(f"Cannot build due to even-build rule on {self.street.name}")

        if current_player.balance < cost_of_one_building:
            logger.warning(
                f"{current_player.name} cannot afford base cost "
                f"${cost_of_one_building} for one {building_type_attempted.name} "
                f"on {self.street.name}."
            )
            return


        is_shortage = False
        threshold_needed = 0
        if building_type_attempted == BuildingType.HOUSE:
            if state.houses_available <= MAX_HOUSES_AVAILABLE_FOR_AUCTION:
                is_shortage = True
            threshold_needed = HOUSE_AUCTION_THRESHOLD
        elif building_type_attempted == BuildingType.HOTEL:
            if state.hotels_available <= MAX_HOTELS_AVAILABLE_FOR_AUCTION:
                is_shortage = True
            threshold_needed = HOTEL_AUCTION_THRESHOLD
        
        if is_shortage:
            competitors = state.get_potential_building_auction_competitors(
                initiator=current_player,
                building_type_to_auction=building_type_attempted,
                cost_basis_property=self.street
            )
            if len(competitors) >= threshold_needed:
                logger.info(f"Building shortage for {building_type_attempted.name} on {self.street.name}. "
                            f"Available: H:{state.houses_available}, HTL:{state.hotels_available}. "
                            f"Competitors: {len(competitors)}. Triggering auction.")
                
                auction_participants = [current_player] + competitors
                try:
                    initial_bidder_idx_in_game_players = state.players.index(current_player)
                except ValueError:
                    logger.error(f"CRITICAL: Initiator {current_player.name} not in state.players. Cannot start auction.")
                    return

                state.auction_state = AuctionState(
                    auction_item=(building_type_attempted, self.street), 
                    participants=auction_participants,
                    bids=[],
                    initial_bidding_index=initial_bidder_idx_in_game_players 
                )
                try:
                    initial_auction_participant_index = auction_participants.index(current_player)
                    state.auction_state.current_bidder_index = initial_auction_participant_index
                except ValueError:
                    logger.error(f"CRITICAL: Initiator {current_player.name} not in auction_participants list. Aborting auction setup.")
                    state.auction_state = None
                    return

                state.property_decision_made_this_landing = True
                return 

        logger.info(f"No building auction triggered for {current_player.name}'s attempt on {self.street.name}. Proceeding with normal build.")


        if building_type_attempted == BuildingType.HOUSE:
            can_build_this_many = True
            for _ in range(self.quantity):
                if not state.player_can_build_on_property(current_player, self.street, BuildingType.HOUSE, check_even_build=True):

                    if self.quantity > 1: logger.warning("Multi-house even build check is simplified here.")
                    pass

                if not can_build_this_many:
                    logger.warning(f"{current_player.name} cannot build {self.quantity} houses on {self.street.name} due to even-build rules for the full quantity.")
                    return

            total_cost = self.street.color_set.house_cost * self.quantity
            if current_player.balance < total_cost:
                logger.warning(f"{current_player.name} cannot afford to build {self.quantity} house(s) for ${total_cost} on {self.street.name}.")
                return
            if state.houses_available < self.quantity:
                logger.warning(f"Not enough houses in bank ({state.houses_available}) to build {self.quantity} for {current_player.name} on {self.street.name}.")
                return
            
            current_player.balance -= total_cost
            state.houses_available -= self.quantity
            self.street.houses += self.quantity
            logger.info(f"{current_player.name} built {self.quantity} house(s) on {self.street.name}. Houses left: {self.street.houses}.")

        elif building_type_attempted == BuildingType.HOTEL:
            if self.street.houses != 4:
                logger.warning(f"{current_player.name} cannot build hotel on {self.street.name}: needs 4 houses first.")
                return
            
            total_cost = self.street.color_set.hotel_cost
            if current_player.balance < total_cost:
                logger.warning(f"{current_player.name} cannot afford to build hotel for ${total_cost} on {self.street.name}.")
                return
            if state.hotels_available < 1:
                logger.warning(f"Not enough hotels in bank ({state.hotels_available}) for {current_player.name} on {self.street.name}.")
                return

            current_player.balance -= total_cost
            state.hotels_available -= 1
            state.houses_available += 4
            self.street.houses = 0
            self.street.hotels = 1
            logger.info(f"{current_player.name} built a hotel on {self.street.name}.")
        
        state.property_decision_made_this_landing = True

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        mask = [False] * (MAX_STREETS * MAX_BUILD_COUNT)
        if state.auction_state or state.pending_trade:
            return mask
        for street in current_player.properties:
            if isinstance(street, Street) and state.player_has_complete_color_set(current_player, street.color_set) and not street.is_mortgaged:
                if street.hotels > 0:
                    continue
                
                same_color_streets = [
                    s for s in state.board.board
                    if isinstance(s, Street) and s.color_set == street.color_set
                ]

                max_build = min(5 - street.houses, state.houses_available)
                if max_build <= 0:
                    continue

                for qty in range(1, max_build + 1):
                    new_total = street.houses + qty

                    can_build = True
                    for other in same_color_streets:
                        if other == street:
                            continue 
                        if other.houses < (new_total - 1):
                            can_build = False
                            break

                    if can_build:
                        idx = street.street_idx * MAX_BUILD_COUNT + (qty - 1)
                        cost_of_build = street.color_set.house_cost * qty
                        if 0 <= idx < len(mask):
                            mask[idx] = current_player.balance >= cost_of_build
                        else:
                            logger.error(f"BuildAction mask index out of range: {idx}")

        return mask


    @classmethod
    def flat_parameter_size(cls) -> int:
        return MAX_STREETS * MAX_BUILD_COUNT

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()

        if state.auction_state and \
        state.auction_state.placing_building_after_win and \
        state.auction_state.building_winner == current_player:
            
            street_param_mask = [False] * MAX_STREETS
            quantity_param_mask = [False] * MAX_BUILD_COUNT
            quantity_param_mask[0] = True
            building_to_place = state.auction_state.building_type_to_place
            
            for street_obj_idx, street_obj in enumerate(state.board.streets): # Iterate all streets
                if street_obj.owner == current_player and \
                state.player_can_build_on_property(current_player, street_obj, building_to_place, check_even_build=True):
                    street_param_mask[street_obj.street_idx] = True
            
            return {
                "street": street_param_mask,
                "quantity": quantity_param_mask,
            }
        
        if state.auction_state or state.pending_trade:
            return {"street": [False] * MAX_STREETS, "quantity": [False] * MAX_BUILD_COUNT}

        pairs = []
        for property_candidate in current_player.properties:
            if not isinstance(property_candidate, Street):
                continue
            street = property_candidate
            if state.player_can_build_on_property(current_player, street, BuildingType.HOUSE):
                if current_player.balance >= street.color_set.house_cost and state.houses_available > 0:
                    pairs.append((street.street_idx, 1))
                    
            if state.player_can_build_on_property(current_player, street, BuildingType.HOTEL):
                if current_player.balance >= street.color_set.hotel_cost and state.hotels_available > 0:
                    pairs.append((street.street_idx, 5))
                    
        street_mask = [False] * MAX_STREETS
        quantity_mask = [False] * MAX_BUILD_COUNT
        
        if not pairs:
            return {"street": street_mask, "quantity": quantity_mask}

        for s_idx, qty in pairs:
            if 0 <= s_idx < MAX_STREETS:
                street_mask[s_idx] = True
            if 1 <= qty <= MAX_BUILD_COUNT:
                quantity_mask[qty - 1] = True

        return {
            "street": street_mask,
            "quantity": quantity_mask,
        }


    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {
            "street": Discrete(MAX_STREETS),
            "quantity": Discrete(MAX_BUILD_COUNT),
        }

    def to_dict(self) -> Dict:
        return {
            "type": "Build",
            "mgn_code": self.player.mgn_code,
            "street_index": self.street.street_idx,
            "tile_index": self.street.index,
            "quantity": self.quantity,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> BuildAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        street = state.board.streets[data["street_index"]]
        return BuildAction(player, street, data["quantity"])

class SellBuildingAction(Action):
    def __init__(self, player: Player, street: Street, quantity: int):
        super().__init__("SellBuilding", player)
        self.street = street
        self.quantity = quantity

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} SH@{self.street.index}x{self.quantity}"

    def process(self, state: State) -> None:
        if self.street.houses >= self.quantity:
            refund = (self.street.color_set.house_cost * self.quantity) // 2
            self.street.sell(self.quantity)
            self.player.balance += refund
            state.houses_available += self.quantity

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        mask = [False] * (MAX_STREETS * MAX_BUILD_COUNT)
        if state.auction_state or state.pending_trade:
            return mask
        for street in current_player.properties:
            if isinstance(street, Street) and street.houses > 0:
                max_sell = min(street.houses, 5)
                for qty in range(1, max_sell + 1):
                    idx = street.street_idx * MAX_BUILD_COUNT + (qty - 1)
                    if 0 <= idx < len(mask):
                        mask[idx] = True
                    else:
                        logger.error(f"SellBuildingAction mask index out of range: {idx}")

        return mask

    @classmethod
    def flat_parameter_size(cls) -> int:
        return MAX_STREETS * MAX_BUILD_COUNT

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        pairs = []
        for street in current_player.properties:
            if state.auction_state or state.pending_trade:
                continue
            if isinstance(street, Street) and street.houses > 0:
                max_sell = min(street.houses, 5)
                for qty in range(1, max_sell + 1):
                    pairs.append((street.street_idx, qty))

        street_mask = [False] * MAX_STREETS
        quantity_mask = [False] * MAX_BUILD_COUNT
        for street_idx, q in pairs:
            if street_idx >= MAX_STREETS or street_idx < 0:
                logger.error(f"Invalid street_idx: {street_idx}")
                continue 
            street_mask[street_idx] = True
            quantity_mask[q - 1] = True

        return {
            "street": street_mask,
            "quantity": quantity_mask,
        }

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {
            "street": Discrete(MAX_STREETS),
            "quantity": Discrete(MAX_BUILD_COUNT),
        }

    def to_dict(self) -> Dict:
        return {
            "type": "SellBuilding",
            "mgn_code": self.player.mgn_code,
            "street_index": self.street.street_idx,
            "quantity": self.quantity,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> SellBuildingAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        street = state.board.streets[data["street_index"]]
        return SellBuildingAction(player, street, data["quantity"])

class UseJailCardAction(Action):
    def __init__(self, player: Player):
        super().__init__("UseJailCard", player)

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} GOOJF"

    def process(self, state: State) -> None:
        if self.player.jail_free_cards > 0 and self.player.in_jail:
            self.player.jail_free_cards -= 1
            self.player.in_jail = False
            self.player.jail_turns = 0

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        return [current_player.jail_free_cards > 0 and current_player.in_jail and not state.pending_trade and not state.auction_state]

    @classmethod 
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        return {"use_card": [current_player.jail_free_cards > 0 and current_player.in_jail and not state.pending_trade and not state.auction_state]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"use_card": Discrete(1)}

    def to_dict(self) -> Dict:
        return {
            "type": "UseJailCard",
            "mgn_code": self.player.mgn_code
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> UseJailCardAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        return UseJailCardAction(player)

class PayJailFineAction(Action):
    def __init__(self, player: Player):
        super().__init__("PayJailFine", player)

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} P${JAIL_BAIL_AMOUNT}"

    def process(self, state: State) -> None:
        if self.player.balance >= JAIL_BAIL_AMOUNT and self.player.in_jail:
            self.player.balance -= JAIL_BAIL_AMOUNT
            self.player.in_jail = False
            self.player.jail_turns = 0

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        return [current_player.balance >= JAIL_BAIL_AMOUNT and current_player.in_jail and not state.pending_trade and not state.auction_state]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        return {"pay_fine": [current_player.balance >= JAIL_BAIL_AMOUNT and current_player.in_jail and not state.pending_trade and not state.auction_state]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"pay_fine": Discrete(1)}


    def to_dict(self) -> Dict:
        return {
            "type": "PayJailFine",
            "mgn_code": self.player.mgn_code,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> PayJailFineAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        return PayJailFineAction(player)

class RollJailAction(Action):
    def __init__(self, player: Player):
        super().__init__("RollJail", player)

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} R"

    def process(self, state: State) -> None:
        if self.player.in_jail and self.player.jail_turns < 3:
            self.player.jail_turns += 1

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        return [current_player.in_jail and current_player.jail_turns < 3 and not state.pending_trade and not state.auction_state]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        current_player = state.current_player()
        return {"roll": [current_player.in_jail and current_player.jail_turns < 3 and not state.pending_trade and not state.auction_state]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"roll": Discrete(1)}


    def to_dict(self) -> Dict:
        return {
            "type": "RollJail",
            "mgn_code": self.player.mgn_code,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> RollJailAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        return RollJailAction(player)

class BankruptcyAction(Action):
    def __init__(self, player: Player):
        super().__init__("Bankruptcy", player)

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} Bankrupt"

    def process(self, state: State) -> None:
        player_to_remove = self.player
        try:
            original_player_list_idx = state.players.index(player_to_remove)
        except ValueError:
            logger.error(f"Critical Error: Player {player_to_remove.name} not found in state.players during bankruptcy processing.")
            original_player_list_idx = state.current_player_index
            if player_to_remove not in state.players:
                 logger.error(f"Player {player_to_remove.name} was not in state.players. Cannot remove.")
                 return


        logger.info(f"Player {player_to_remove.name} is declaring bankruptcy.")
        for prop in player_to_remove.properties:
            prop.owner = None
            if isinstance(prop, Street):
                if prop.hotels > 0:
                    state.hotels_available += prop.hotels
                    prop.hotels = 0
                if prop.houses > 0:
                    state.houses_available += prop.houses
                    prop.houses = 0
            prop.is_mortgaged = False

        player_to_remove.properties.clear()
        player_to_remove.is_bankrupt = True # Mark player as bankrupt

        # Remove player from the game list
        if player_to_remove in state.players:
            state.players.pop(original_player_list_idx)
        else:
            logger.warning(f"Player {player_to_remove.name} was already removed or not found when trying to pop.")


        logger.info(f"Player {player_to_remove.name} has been removed from the game.")

        if len(state.players) <= 1:
            logger.info("Game is ending due to bankruptcy reducing players to 1 or 0.")
            if len(state.players) == 1:
                state.current_player_index = 0
            return

        if original_player_list_idx < state.current_player_index:
            state.current_player_index -= 1
        state.current_player_index %= len(state.players)

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        current_player = state.current_player()
        return [current_player.balance < 0]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        valid = None
        current_player = state.current_player()
        if VOLUNTARY_BANKRUPTCY:
            valid = current_player.balance < 0
        else:
            current_player
        return {"bankrupt": [valid]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"bankrupt": Discrete(1)}

    def to_dict(self) -> Dict:
        return {
            "type": "Bankruptcy",
            "mgn_code": self.player.mgn_code,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> BankruptcyAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        return BankruptcyAction(player)

class AuctionFoldAction(Action):
    def __init__(self, player: Player):
        super().__init__("AuctionFold", player)

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code}.F"

    def process(self, state: State) -> None:
        #print("Processing auction fold action")
        if state.auction_state and self.player in state.auction_state.participants:
            #print(f"Players before={state.auction_state.to_dict()}")
            state.auction_state.participants.remove(self.player)
            #print(f"Players after={state.auction_state.to_dict()}")
        else:
            logger.warning(f"{self.player.name} is not a participant in the current auction.")

        if state.auction_state.is_done():
            logger.info("Auction is done after folding.")
            state.auction_state.resolve(state=state)
        else:
            #print(f"Auctioning is not done. proceeding. auction_state={state.auction_state.to_dict()}")
            state.advance_auction_turn(player=self.player)
            #print(f"Advanced turn. new auction_state={state.auction_state.to_dict()}")

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        valid = bool(state.auction_state and len(state.auction_state.participants) >= 1)
        return [valid]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        valid = bool(state.auction_state and len(state.auction_state.participants) >= 1)
        return {"fold": [valid]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"fold": Discrete(1)}

    def to_dict(self) -> Dict:
        return {"type": "AuctionFold", "mgn_code": self.player.mgn_code}

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> AuctionFoldAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        return AuctionFoldAction(player)

class AcceptTradeAction(Action):
    """If a trade is pending, and the current player is the responder, accept it."""
    def __init__(self, player: Player):
        super().__init__("AcceptTrade", player)

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} ACCEPT"

    def process(self, state: State) -> None:
        if not state.pending_trade:
            logger.warning("No pending trade to accept.")
            return

        trade = state.pending_trade

        if trade.responder != self.player:
            logger.warning(f"{self.player.name} is not the responder for this pending trade.")
            return

        logger.info(f"{self.player.name} accepts trade from {trade.proposer.name}")
        self._execute_trade(state, trade)

        state.pending_trade = None

    def _execute_trade(self, state: State, trade: TradeOffer):
        proposer = trade.proposer
        responder = trade.responder

        if proposer.balance < trade.cash_offered:
            logger.error(f"Trade failed: Proposer {proposer.name} lacks cash ${trade.cash_offered}")
            return
        if proposer.jail_free_cards < trade.get_out_of_jail_cards_offered:
             logger.error(f"Trade failed: Proposer {proposer.name} lacks jail cards ({trade.get_out_of_jail_cards_offered})")
             return
        for prop in trade.properties_offered:
            if prop not in proposer.properties:
                logger.error(f"Trade failed: Proposer {proposer.name} does not own {prop.name}")
                return

        if responder.balance < trade.cash_asking:
            logger.error(f"Trade failed: Responder {responder.name} lacks cash ${trade.cash_asking}")
            return
        if responder.jail_free_cards < trade.get_out_of_jail_cards_asking:
             logger.error(f"Trade failed: Responder {responder.name} lacks jail cards ({trade.get_out_of_jail_cards_asking})")
             return
        for prop in trade.properties_asking:
             if prop not in responder.properties:
                 logger.error(f"Trade failed: Responder {responder.name} does not own {prop.name}")
                 return

        logger.debug(f"Executing trade: {trade.to_dict()}")

        proposer.balance -= trade.cash_offered
        responder.balance += trade.cash_offered
        responder.balance -= trade.cash_asking
        proposer.balance += trade.cash_asking

        proposer.jail_free_cards -= trade.get_out_of_jail_cards_offered
        responder.jail_free_cards += trade.get_out_of_jail_cards_offered
        responder.jail_free_cards -= trade.get_out_of_jail_cards_asking
        proposer.jail_free_cards += trade.get_out_of_jail_cards_asking

        for prop in list(trade.properties_offered):
            if prop in proposer.properties:
                proposer.properties.remove(prop)
                responder.properties.append(prop)
                prop.owner = responder
            else:
                 logger.warning(f"Property {prop.name} already removed from {proposer.name} during trade?")

        for prop in list(trade.properties_asking):
            if prop in responder.properties:
                responder.properties.remove(prop)
                proposer.properties.append(prop)
                prop.owner = proposer
            else:
                 logger.warning(f"Property {prop.name} already removed from {responder.name} during trade?")

        logger.info(f"Trade completed successfully between {proposer.name} and {responder.name}.")

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        if not state.pending_trade:
            return [False]
        current_player = state.current_player()
        valid = (current_player == state.pending_trade.responder)

        trade = state.pending_trade
        
        if current_player.balance < trade.cash_asking:
            valid = False
        if current_player.jail_free_cards < trade.get_out_of_jail_cards_asking:
            valid = False
        for prop_asked in trade.properties_asking:
            if prop_asked not in current_player.properties:
                valid = False
                break

        return [valid]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        if not state.pending_trade:
            return {"accept": [False]}
        current_player = state.current_player()
        valid = (current_player == state.pending_trade.responder)

        trade = state.pending_trade
        
        if current_player.balance < trade.cash_asking:
            valid = False
        if current_player.jail_free_cards < trade.get_out_of_jail_cards_asking:
            valid = False
        for prop_asked in trade.properties_asking:
            if prop_asked not in current_player.properties: # Check ownership
                valid = False
                break

        return {"accept": [valid]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"accept": Discrete(1)}

    def to_dict(self) -> Dict:
        return {
            "type": "AcceptTrade",
            "mgn_code": self.player.mgn_code,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> AcceptTradeAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        return AcceptTradeAction(player)



class RejectTradeAction(Action):
    def __init__(self, player: Player):
        super().__init__("RejectTrade", player)

    def to_mgn(self) -> str:
        return f"{self.player.mgn_code} REJECT"

    def process(self, state: State) -> None:
        if not state.pending_trade:
            logger.warning("No pending trade to reject.")
            return

        trade = state.pending_trade
        if trade.responder != self.player:
            logger.warning(f"{self.player.name} is not the responder for this pending trade.")
            return

        logger.info(f"{self.player.name} rejects trade from {trade.proposer.name}")
        state.pending_trade = None

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        if not state.pending_trade:
            return [False]
        return [state.current_player() == state.pending_trade.responder]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str, List[bool]]:
        if not state.pending_trade:
            return {"reject": [False]}
        valid = (state.current_player() == state.pending_trade.responder)
        return {"reject": [valid]}

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {"reject": Discrete(1)}

    def to_dict(self) -> Dict:
        return {
            "type": "RejectTrade",
            "mgn_code": self.player.mgn_code,
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> RejectTradeAction:
        player = next(p for p in state.players if p.mgn_code == data["mgn_code"])
        return RejectTradeAction(player)


class SendMessageAction(Action):
    def __init__(self, sender: Player, message: str, recipient: Optional[Player] = None):
        super().__init__("SendMessage", sender)
        self.message = message
        self.recipient = recipient  # None means public

    def to_mgn(self) -> str:
        if self.recipient:
            return f"{self.player.mgn_code} MSG->{self.recipient.mgn_code}: {self.message}"
        else:
            return f"{self.player.mgn_code} MSG->ALL: {self.message}"

    def process(self, state: State) -> None:
        # Store message in chat log
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "from_mgn_code": self.player.mgn_code,
            "from_name": self.player.name,
            "message": self.message,
            "timestamp": ts
        }
        if self.recipient:
            log_entry["to_mgn_code"] = self.recipient.mgn_code
            log_entry["to_name"] = self.recipient.name
            log_entry["private"] = True
            logger.info(f"[MSG PRIVATE] {self.player.name} to {self.recipient.name}: {self.message}")
        else:
            log_entry["to_mgn_code"] = "ALL"
            log_entry["to_name"] = "ALL"
            log_entry["private"] = False
            logger.info(f"[MSG PUBLIC] {self.player.name}: {self.message}")
        
        state.chat_log.append(log_entry)

    @classmethod
    def to_action_mask_flat(cls, state: State) -> List[bool]:
        return [True]

    @classmethod
    def flat_parameter_size(cls) -> int:
        return 1

    @classmethod
    def to_action_mask_hierarchical(cls, state: State) -> Dict[str,List[bool]]:
        if state.auction_state or state.pending_trade:
            return {
                "recipient": [False] * (MAX_PLAYERS + 1),
                "message": [False]
            }
        
        recipient_mask = [False] * (MAX_PLAYERS + 1)
        recipient_mask[0] = True
        for i in range(len(state.players)):
            if i < MAX_PLAYERS :
                 recipient_mask[i+1] = True 
        
        return {
            "recipient": recipient_mask,
            "message": [True]
        }

    @classmethod
    def hierarchical_parameters(cls) -> Dict[str, Space]:
        return {
        "recipient": Discrete(MAX_PLAYERS+1),
        "message": Box(0, 255, shape=(MAX_MESSAGE_LENGTH,), dtype=np.int32)
        }

    def to_dict(self) -> Dict:
        return {
            "type": "SendMessage",
            "sender_mgn_code": self.player.mgn_code,
            "recipient_mgn_code": self.recipient.mgn_code if self.recipient else "ALL",
            "message": self.message
        }

    @classmethod
    def from_dict(cls, data: Dict, state: State) -> SendMessageAction:
        sender = next(p for p in state.players if p.mgn_code == data["sender_mgn_code"])
        recipient = None
        if data["recipient_mgn_code"] != "ALL":
            recipient = next((p for p in state.players if p.mgn_code == data["recipient_mgn_code"]), None)
            if recipient is None:
                logger.warning(f"SendMessage from_dict: recipient {data['recipient_mgn_code']} not found. Message will be public.")
        
        return SendMessageAction(sender=sender, message=data["message"], recipient=recipient)

HIERARCHICAL_ACTION_CLASSES = [
    RollDiceAction,
    EndTurnAction,
    BuyAction,
    AuctionAction,
    AuctionBidAction,
    MortgageAction,
    UnmortgageAction,
    BuildAction,
    SellBuildingAction,
    UseJailCardAction,
    PayJailFineAction,
    RollJailAction,
    BankruptcyAction,
    AuctionFoldAction,
    ProposeTradeAction,
    AcceptTradeAction,
    RejectTradeAction,
    SendMessageAction
]

HIERARCHICAL_ACTION_CLASSES_WO_SEND_MESSAGE_ACTION = [
    RollDiceAction,
    EndTurnAction,
    BuyAction,
    AuctionAction,
    AuctionBidAction,
    MortgageAction,
    UnmortgageAction,
    BuildAction,
    SellBuildingAction,
    UseJailCardAction,
    PayJailFineAction,
    RollJailAction,
    BankruptcyAction,
    AuctionFoldAction,
    ProposeTradeAction,
    AcceptTradeAction,
    RejectTradeAction,
]

class ActionManager:
    def __init__(
        self,
        action_space_type: ActionSpaceType = ActionSpaceType.HIERARCHICAL,
        include_send_message_action: bool = True
    ):
        self.action_space_type = action_space_type
        if action_space_type == ActionSpaceType.FLAT:
            if include_send_message_action:
                self.action_classes = [c for c in HIERARCHICAL_ACTION_CLASSES]
            else:
                self.action_classes = [c for c in HIERARCHICAL_ACTION_CLASSES_WO_SEND_MESSAGE_ACTION]
        else:
            if include_send_message_action:
                self.action_classes = HIERARCHICAL_ACTION_CLASSES
            else:
                self.action_classes = HIERARCHICAL_ACTION_CLASSES_WO_SEND_MESSAGE_ACTION
            
        self.flat_offsets = self._calculate_flat_offsets()
        self.parameter_spaces = {
            cls.__name__: GymDict(cls.hierarchical_parameters())
            for cls in self.action_classes
        }

    def _calculate_flat_offsets(self) -> Dict[Type[Action], int]:
        offsets = {}
        current = 0
        for cls in self.action_classes:
            offsets[cls] = current
            current += cls.flat_parameter_size()
        return offsets

    def to_action_space(self) -> Space:
        if self.action_space_type == ActionSpaceType.FLAT:
            return Discrete(sum(cls.flat_parameter_size() for cls in self.action_classes))
        else:
            return GymDict({
                "action_type": Discrete(len(self.action_classes)),
                "parameters": GymDict({
                    cls.__name__: space
                    for cls, space in self.parameter_spaces.items()
                })
            })

    def to_action_mask(self, state: State) -> Union[np.ndarray, Dict]:
        if self.action_space_type == ActionSpaceType.FLAT:
            return self._to_action_mask_flat(state)
        else:
            return self._to_action_mask_hierarchical(state)

    def _to_action_mask_flat(self, state: State) -> np.ndarray:
        mask = []
        for cls in self.action_classes:
            mask.extend(cls.to_action_mask_flat(state))
        return np.array(mask, dtype=np.bool_)

    def _to_action_mask_hierarchical(self, state: State) -> Dict:
        action_type_mask = []
        parameters_mask = {cls.__name__: {} for cls in self.action_classes}

        for i, cls in enumerate(self.action_classes):
            cls_mask = cls.to_action_mask_hierarchical(state)
            action_valid = any(any(v) if isinstance(v, list) else v for v in cls_mask.values()) if cls_mask else False
            action_type_mask.append(action_valid)
            parameters_mask[cls.__name__] = cls_mask

        return {
            "action_type": action_type_mask,
            "parameters": parameters_mask
        }

    def decode_action(self, action: Union[int, Dict], state: State) -> Action:
        if self.action_space_type == ActionSpaceType.FLAT:
            return self._decode_flat_action(action, state)
        else:
            return self._decode_hierarchical_action(action, state)

    def _decode_flat_action(self, action_idx: int, state: State) -> Action:
        for cls in self.action_classes:
            offset = self.flat_offsets[cls]
            size = cls.flat_parameter_size()
            if offset <= action_idx < offset + size:
                return self._instantiate_flat_action(cls, action_idx - offset, state)
        raise ValueError(f"Invalid action index: {action_idx}")

    def _decode_hierarchical_action(self, action_dict: Dict, state: State) -> Action:
        action_type_idx = action_dict["action_type"]
        if not (0 <= action_type_idx < len(self.action_classes)):
            raise ValueError(f"Invalid action_type index: {action_type_idx}")
        
        cls = self.action_classes[action_type_idx]
        parameters = action_dict["parameters"][cls.__name__]
        return self._instantiate_hierarchical_action(cls, parameters, state)

    def _get_current_player(self, state: State) -> Player:
        if state.auction_state:
            return state.auction_state.current_participant()
        else:
            return state.current_player()

    def _instantiate_flat_action(self, cls: Type[Action], param: int, state: State) -> Action:
        current_player = self._get_current_player(state)

        if cls == EndTurnAction:
            return EndTurnAction(current_player)
        elif cls == BuyAction:
            return BuyAction(current_player, state.board.board[current_player.position])
        elif cls == AuctionAction:
            return AuctionAction(current_player, state.board.board[current_player.position])
        elif cls == AuctionBidAction:
            return AuctionBidAction(current_player, param)
        elif cls == MortgageAction:
            return MortgageAction(current_player, state.board.board[param])
        elif cls == UnmortgageAction:
            return UnmortgageAction(current_player, state.board.board[param])
        elif cls == BuildAction:
            street_idx = param // MAX_BUILD_COUNT
            quantity = (param % MAX_BUILD_COUNT) + 1
            if 0 <= street_idx < len(state.board.streets):
                street = state.board.streets[street_idx] 
                return BuildAction(current_player, street, quantity)
            else:
                raise ValueError(f"Invalid street_idx: {street_idx}")
        elif cls == SellBuildingAction:
            street_idx = param // MAX_BUILD_COUNT
            quantity = (param % MAX_BUILD_COUNT) + 1
            if 0 <= street_idx < len(state.board.streets):
                street = state.board.streets[street_idx]
                return SellBuildingAction(current_player, street, quantity)
            else:
                raise ValueError(f"Invalid street_idx: {street_idx}")
        elif cls == UseJailCardAction:
            return UseJailCardAction(current_player)
        elif cls == PayJailFineAction:
            return PayJailFineAction(current_player)
        elif cls == RollJailAction:
            return RollJailAction(current_player)
        elif cls == BankruptcyAction:
            return BankruptcyAction(current_player)
        elif cls == RollDiceAction:
            return RollDiceAction(current_player)
        elif cls == AuctionFoldAction:
            return AuctionFoldAction(current_player)
        else:
            raise NotImplementedError(f"Flat decoding not implemented for {cls.__name__}")

    def _instantiate_hierarchical_action(self, cls: Type[Action], params: Dict, state: State) -> Action:
        current_player = self._get_current_player(state)

        if cls == EndTurnAction:
            return EndTurnAction(current_player)
        elif cls == BuyAction:
            return BuyAction(current_player, state.board.board[current_player.position])
        elif cls == AuctionAction:
            return AuctionAction(current_player, state.board.board[current_player.position])
        elif cls == AuctionBidAction:
            return AuctionBidAction(current_player, params["bid_amount"])
        elif cls == MortgageAction:
            return MortgageAction(current_player, state.board.properties[params["property"]])
        elif cls == UnmortgageAction:
            return UnmortgageAction(current_player, state.board.properties[params["property"]])
        elif cls == BuildAction:
            street_idx = params["street"]
            quantity = params["quantity"] + 1 
            if 0 <= street_idx < len(state.board.streets):
                street = state.board.streets[street_idx]
                return BuildAction(current_player, street, quantity)
            else:
                raise ValueError(f"Invalid street_idx: {street_idx}")
        elif cls == SellBuildingAction:
            street_idx = params["street"]
            quantity = params["quantity"] + 1 
            if 0 <= street_idx < len(state.board.streets):
                street = state.board.streets[street_idx]
                return SellBuildingAction(current_player, street, quantity)
            else:
                raise ValueError(f"Invalid street_idx: {street_idx}")
        elif cls == UseJailCardAction:
            return UseJailCardAction(current_player)
        elif cls == PayJailFineAction:
            return PayJailFineAction(current_player)
        elif cls == RollJailAction:
            return RollJailAction(current_player)
        elif cls == BankruptcyAction:
            return cls(current_player)
        elif cls == AuctionFoldAction:
            return AuctionFoldAction(current_player)
        elif cls == RollDiceAction:
            return RollDiceAction(current_player)
        elif cls == ProposeTradeAction:
            responder_idx = params["trade_partner"]
            possible_responders = [p for p in state.players if p != current_player]
            if responder_idx < 0 or responder_idx >= len(possible_responders):
                raise ValueError(f"Invalid trade_partner index: {responder_idx} for {len(possible_responders)} possible partners")

            partner_player = possible_responders[responder_idx]

            cash_offered = params["cash_offered"]
            properties_offered_indices = [i for i, owned in enumerate(params["properties_offered"]) if owned == 1]
            properties_offered = [prop for prop in current_player.properties if prop.property_idx in properties_offered_indices]

            get_out_of_jail_cards_offered = params["get_out_of_jail_cards_offered"]

            cash_asking = params["cash_asking"]
            properties_asking_indices = [i for i, owned in enumerate(params["properties_asking"]) if owned == 1]
            properties_asking = [prop for prop in partner_player.properties if prop.property_idx in properties_asking_indices]

            get_out_of_jail_cards_asking = params["get_out_of_jail_cards_asking"]

            return ProposeTradeAction(
                trade_offer=TradeOffer(
                    proposer=current_player,
                    responder=partner_player,
                    cash_offered=cash_offered,
                    properties_offered=properties_offered,
                    get_out_of_jail_cards_offered=get_out_of_jail_cards_offered,
                    cash_asking=cash_asking,
                    properties_asking=properties_asking, 
                    get_out_of_jail_cards_asking=get_out_of_jail_cards_asking
                )
            )
        elif cls == AcceptTradeAction:
            return AcceptTradeAction(current_player)
        elif cls == RejectTradeAction:
            return RejectTradeAction(current_player)
        elif cls == SendMessageAction:
            sender = current_player
            recipient_param_val = params["recipient"]
            message_int_array = params["message"]    

            actual_bytes = message_int_array[message_int_array != 0].astype(np.uint8)
            try:
                decoded_message_str = actual_bytes.tobytes().decode('utf-8')
            except UnicodeDecodeError:
                decoded_message_str = actual_bytes.tobytes().decode('latin-1', errors='replace')
                logger.warning(f"Message from {sender.name} had encoding issues. Used fallback.")
            
            recipient_player_obj: Optional[Player] = None
            if recipient_param_val == 0:
                recipient_player_obj = None
            elif 1 <= recipient_param_val <= len(state.players):
                recipient_player_obj = state.players[recipient_param_val - 1]
                if recipient_player_obj == sender:
                     logger.debug(f"{sender.name} is sending a message to themselves.")
            else:
                logger.error(f"Invalid recipient index {recipient_param_val} for SendMessageAction. Defaulting to public.")
                recipient_player_obj = None 
            
            return SendMessageAction(sender=sender, message=decoded_message_str, recipient=recipient_player_obj)
        else:
            raise NotImplementedError(f"Hierarchical decoding not implemented for {cls.__name__}")