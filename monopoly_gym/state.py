# monopoly_gym/gym/state.py
from __future__ import annotations
from enum import Enum
from typing import List, Literal, Optional, Tuple, TYPE_CHECKING, Union
from monopoly_gym.board import Board
from monopoly_gym.tile import Chance, CommunityChest, Property, ColorSet, Railroad, SpecialTile, SpecialTileType, Street, Tax, Utility
from gym.spaces import Dict, Discrete, Box
import logging

if TYPE_CHECKING:
    from monopoly_gym.player import Player

# How many *other* players (besides the one who just tried to build) must be
# eligible/willing to trigger a house‐auction or hotel‐auction?
HOUSE_AUCTION_THRESHOLD = 2
HOTEL_AUCTION_THRESHOLD = 2

# At or below THIS many available houses (or hotels) triggers the shortage auction
MAX_HOUSES_AVAILABLE_FOR_AUCTION = 1
MAX_HOTELS_AVAILABLE_FOR_AUCTION = 1


logger = logging.getLogger(__name__)

class BuildingType(Enum):
    HOUSE = "house"
    HOTEL = "hotel"

    @property
    def name(self):
        return self.value

class TradeOffer:
    def __init__(self, proposer: Player, responder: Player, cash_offered: int, properties_offered: List[Property], get_out_of_jail_cards_offered: int,  cash_asking: int, properties_asking:  List[Property], get_out_of_jail_cards_asking: int ):
        self.proposer = proposer
        self.responder = responder
        # offer
        self.cash_offered = cash_offered
        self.properties_offered = properties_offered
        self.get_out_of_jail_cards_offered = get_out_of_jail_cards_offered
        # asking
        self.cash_asking = cash_asking
        self.properties_asking = properties_asking
        self.get_out_of_jail_cards_asking = get_out_of_jail_cards_asking

    def to_dict(self) -> dict:
        return {
            "type": "ProposeTrade",
            "proposer": self.proposer.mgn_code,
            "responder": self.responder.mgn_code,
            "cash_offered": self.cash_offered,
            "properties_offered_indices": [p.property_idx for p in self.properties_offered],
            "get_out_of_jail_cards_offered": self.get_out_of_jail_cards_offered,
            "cash_asking": self.cash_asking,
            "properties_asking_indices": [p.property_idx for p in self.properties_asking],
            "get_out_of_jail_cards_asking": self.get_out_of_jail_cards_asking,
        }

class AuctionBid:
    def __init__(self, bidder: Player, bid_amount: int):
        self.bidder = bidder
        self.bid_amount = bid_amount

    def to_dict(self) -> dict:
        return {
            "bidder": self.bidder.name,  # Serialize the player's name
            "bid_amount": self.bid_amount
        }

class AuctionState:
    def __init__(self, aucition_item: Union[Property, Tuple[BuildingType, Street]], participants:  List[Player], bids: List[AuctionBid], initial_bidding_index: int = 0):
        self.auction_item = aucition_item
        self.participants = participants
        self.bids = bids
        self.current_bidder_index: int = initial_bidding_index
        self.placing_building_after_win: bool = False 


    def is_done(self):
        if (len(self.participants) == 1 and len(self.bids) > 0) or len(self.participants) == 0:
            return True
        return False
    
    def highest_bid(self) -> Optional[AuctionBid]:
        if len(self.bids) == 0:
            return None
        return max(self.bids, key=lambda bid: bid.bid_amount)

    def resolve(self, state: State):
        if len(self.bids) == 0:
            if isinstance(self.auction_item, Property):
                logger.info(f"Auction for {self.auction_item.name} ended with no bids. Property remains unowned.")
            elif isinstance(self.auction_item, tuple):
                building_type_auctioned = self.auction_item[0]
                logger.info(f"Auction for {building_type_auctioned.name} ended with no bids.")
            state.auction_state = None
            state.property_decision_made_this_landing = True
            return

        if not self.is_done():
            logger.warning("Auction resolve called but auction is not done.")
            return

        highest_bid_obj = self.highest_bid()

        if highest_bid_obj is None:
            # No bids were placed, even if there was only one participant who then folded (or no one bid)
            if isinstance(self.auction_item, Property):
                logger.info(f"Auction for {self.auction_item.name} ended with no bids. Property remains unowned.")
            elif isinstance(self.auction_item, tuple):
                building_type_auctioned: BuildingType = self.auction_item[0]
                logger.info(f"Auction for {building_type_auctioned.name} ended with no bids.")
            state.auction_state = None
            state.property_decision_made_this_landing = True # Decision (to auction) was made.
            return

        auction_winner = highest_bid_obj.bidder
        bid_amount = highest_bid_obj.bid_amount

        if isinstance(self.auction_item, Property):
            logger.info(f"{auction_winner.name} won the auction for {self.auction_item.name} at ${bid_amount}.")
            if auction_winner.balance < bid_amount:
                logger.error(f"CRITICAL: {auction_winner.name} won auction but cannot afford bid ${bid_amount}. State: {auction_winner.balance}")
            auction_winner.balance -= bid_amount
            auction_winner.properties.append(self.auction_item)
            self.auction_item.owner = auction_winner
            state.auction_state = None
            state.property_decision_made_this_landing = True

        elif isinstance(self.auction_item, tuple): # Building auction
            building_type_won: BuildingType = self.auction_item[0]

            logger.info(f"{auction_winner.name} won auction for one {building_type_won.name} at ${bid_amount}.")

            if auction_winner.balance < bid_amount:
                logger.error(f"CRITICAL: {auction_winner.name} won building auction but cannot afford bid ${bid_amount}. State: {auction_winner.balance}")
                state.auction_state = None
                return

            # Deduct payment
            auction_winner.balance -= bid_amount

            # Update bank inventory
            if building_type_won == BuildingType.HOUSE:
                if state.houses_available < 1:
                    logger.error(f"CRITICAL: {auction_winner.name} won house auction but no houses available in bank post-bidding! Refunding bid.")
                    auction_winner.balance += bid_amount # Refund
                    state.auction_state = None
                    return
                state.houses_available -= 1
            elif building_type_won == BuildingType.HOTEL:
                if state.hotels_available < 1:
                    logger.error(f"CRITICAL: {auction_winner.name} won hotel auction but no hotels available post-bidding! Refunding bid.")
                    auction_winner.balance += bid_amount # Refund
                    state.auction_state = None
                    return
                state.hotels_available -= 1
                # Note: The 4 houses return to the bank when the hotel is placed, not here.

            self.placing_building_after_win = True
            
            # The game's current player must become the auction winner for the next action
            try:
                state.current_player_index = state.players.index(auction_winner)
            except ValueError:
                logger.error(f"Auction winner {auction_winner.name} not found in state.players. Critical error. Building not placed.")
                # Revert payment and bank inventory
                auction_winner.balance += bid_amount
                if building_type_won == BuildingType.HOUSE: state.houses_available += 1
                elif building_type_won == BuildingType.HOTEL: state.hotels_available += 1
                state.auction_state = None
                return
            
            state.rolled_this_turn = True
            state.property_decision_made_this_landing = True 
            
            logger.info(f"{auction_winner.name} (now current player) must place the won {building_type_won.name}.")
        else:
            logger.error(f"Unknown auction item type in resolve: {type(self.auction_item)}")
            state.auction_state = None

    def current_participant(self) -> Player:
        """Return the current player."""
        if not self.participants:
            raise ValueError("No players in the game")
        return self.participants[self.current_bidder_index]

    def to_dict(self) -> dict:
        return {
            "auction_item": self.auction_item.name,
            "participants": [p.name for p in self.participants],
            "bids": [bid.to_dict() for bid in self.bids],
            "current_bidder_index": self.current_bidder_index
        }

def eligible_building_bidders(state: "State", property_idx: int, building_type: str) -> List[int]:
    bidders: List[int] = []
    tile = state.board[property_idx]
    if not isinstance(tile, Street):
        return []

    color_group = tile.color_set
    group_props = [i for i, t in enumerate(state.board) 
                   if getattr(t, "color_group", None) == color_group]

    for p_idx, p in enumerate(state.players):
        owned_indices = {prop.index for prop in p.properties if prop.type == "Street" and prop.color_group == color_group}
        if set(group_props) != owned_indices:
            continue

        my_prop = next((prop for prop in p.properties if prop.index == property_idx), None)
        if my_prop is None:
            continue

        if building_type == "house":
            if my_prop.houses >= 4:
                continue
            cost = tile.house_cost
        else:
            if my_prop.hotels >= 1 or my_prop.houses != 4:
                continue
            cost = tile.hotel_cost

        if p.balance >= cost:
            bidders.append(p_idx)

    return bidders


class State:
    def __init__(self, max_turns=50, logger: logging.Logger=None):
        self.board = Board(houses_available=32, hotels_available=12)
        self.players : List[Player] = []
        self.current_player_index: int = 0
        self.current_consecutive_doubles: int = 0
        self.max_turns: int = max_turns
        self.turn_counter: int = 0
        self.houses_available: int = 32
        self.hotels_available: int = 12
        self.auction_state: Optional[AuctionState] = None
        self.pending_trade: Optional[TradeOffer] = None
        self.rolled_this_turn = False
        self.chat_log: List[Dict] = []
        self.last_dice_roll: Optional[Tuple[int, int]] = None
        self.pending_debt_amount: Optional[int] = None
        self.pending_creditor: Optional[Union[Player, Literal["Bank"]]] = None
        self.property_decision_made_this_landing: bool = False
        self.logger = logger

    def advance_turn(self, player: Player) -> int:
        if player in self.players:
            if player.balance < 0:
                return None
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
        elif self.current_player_index >= len(self.players):
            self.current_player_index = 0

    def advance_auction_turn(self, player: Player) -> int:
        if player in self.auction_state.participants:
            self.auction_state.current_bidder_index = (self.auction_state.current_bidder_index + 1) % len(self.auction_state.participants)
        elif self.auction_state.current_bidder_index >= len(self.auction_state.participants):
            self.auction_state.current_bidder_index = 0

    def to_dict(self) -> dict:
        return {
            "board": [tile.to_dict() for tile in self.board.board],
            "players": [player.to_dict() for player in self.players],
            "current_player_index": self.current_player_index,
            "current_consecutive_doubles": self.current_consecutive_doubles,
            "max_turns": self.max_turns,
            "turn_counter": self.turn_counter,
            "houses_available": self.houses_available,
            "hotels_available": self.hotels_available,
            "auction_state": self.auction_state.to_dict() if self.auction_state else None
        }
    
    def to_observation(self) -> Dict:
        return {
            "houses_available": self.houses_available,
            "hotels_available": self.hotels_available,
            "current_player_index": self.current_player_index,
            "turn_counter": self.turn_counter,
            "auction_state": 1 if self.auction_state is not None else 0,
            "players": [
                {
                    "balance": player.balance,
                    "position": player.position,
                    "properties": [prop.index for prop in player.properties],
                    "in_jail": int(player.in_jail),
                }
                for player in self.players
            ],
            "properties": [
                {
                    "owner": prop.owner.mgn_code if prop.owner else None,
                    "is_mortgaged": int(prop.is_mortgaged),
                }
                for prop in self.board.board
                if isinstance(prop, Property)
            ],
        }

    def reset(self):
        self.board = Board(houses_available=32, hotels_available=12)
        self.players = []
        self.current_player_index = 0
        self.current_consecutive_doubles = 0
        self.turn_counter = 0
        self.houses_available = 32
        self.hotels_available = 12
        self.auction_state = None
        self.rolled_this_turn = False
        return self 

    def player_has_complete_color_set(self, player: Player, color_set: ColorSet) -> bool:
        color_properties = [prop for prop in self.board.board if isinstance(prop, Street) and prop.color_set == color_set]
        owns_full_set = all(prop.owner == player for prop in color_properties)
        return owns_full_set


    def current_player(self) -> Player:
        """Return the current player."""
        if self.auction_state:
            return self.auction_state.current_participant()
        if self.pending_trade:
            return self.pending_trade.responder
        if not self.players:
            raise ValueError("No players in the game")
        try:
            return self.players[self.current_player_index]
        except Exception as ex:
            print(f"Failed to set players={self.players} of len={len(self.players)} with ex={str(ex)} and current_player_index={self.current_player_index}")
            return self.players[self.current_player_index]

    def send_player_to_jail(self, player: Player):
        player.position = 10
        player.in_jail = True
        self.current_consecutive_doubles = 0
        self.advance_turn(player=player)


    def handle_landing_on_tile(self, player: Player, dice_roll: Tuple[int, int]):
        current_tile = self.board.board[player.position]
        if self.logger is not None:
            self.logger.info(f"{player.name} landed on {current_tile.name}.")
        self.property_decision_made_this_landing = False
        if isinstance(current_tile, Property):
            if current_tile.owner is None:
                if self.logger is not None:
                    self.logger.info(f"{current_tile.name} is available for purchase at ${current_tile.purchase_cost}.")
            elif current_tile.owner != player:
                rent = self.calculate_rent(property=current_tile, dice_roll=dice_roll)
                if self.logger is not None:
                    self.logger.info(f"{player.name} landed on {current_tile.name}, owned by {current_tile.owner.name}. Rent is ${rent}.")
                self.pending_creditor = current_tile.owner
                self.pending_debt_amount = rent
                player.balance -= rent
                current_tile.owner.balance += rent
        elif isinstance(current_tile, Tax):
            self.pending_creditor = "Bank"
            self.pending_debt_amount = current_tile.tax_amount
            player.balance -= current_tile.tax_amount
        elif isinstance(current_tile, Chance):
            # Draw the Chance card
            card_id, card_text, card_effect = self.board.chance_cards.pop()
            if self.logger is not None:
                self.logger.info(f"{player.name} draws Chance #{card_id}: “{card_text}”")

            before = {
                "pos": player.position,
                "bal": player.balance,
                "in_jail": player.in_jail,
                "jail_cards": player.jail_free_cards,
            }

            card_effect(self)

            after = {
                "pos": player.position,
                "bal": player.balance,
                "in_jail": player.in_jail,
                "jail_cards": player.jail_free_cards,
            }

            self.logger.info(
                f"Chance effect → pos {before['pos']}→{after['pos']}, "
                f"bal {before['bal']}→{after['bal']}, "
                f"in_jail {before['in_jail']}→{after['in_jail']}, "
                f"jail_cards {before['jail_cards']}→{after['jail_cards']}"
            )

            if card_id != 7:
                self.board.chance_cards.insert(0, (card_id, card_text, card_effect))

        elif isinstance(current_tile, CommunityChest):
            card_id, card_text, card_effect = self.board.community_chest_cards.pop()
            self.logger.info(f"{player.name} draws CC #{card_id}: “{card_text}”")

            before = {
                "pos": player.position,
                "bal": player.balance,
                "in_jail": player.in_jail,
                "jail_cards": player.jail_free_cards,
            }

            card_effect(self)

            after = {
                "pos": player.position,
                "bal": player.balance,
                "in_jail": player.in_jail,
                "jail_cards": player.jail_free_cards,
            }

            self.logger.info(
                f"CC effect → pos {before['pos']}→{after['pos']}, "
                f"bal {before['bal']}→{after['bal']}, "
                f"in_jail {before['in_jail']}→{after['in_jail']}, "
                f"jail_cards {before['jail_cards']}→{after['jail_cards']}"
            )

            if card_id != 5:
                self.board.community_chest_cards.insert(0, (card_id, card_text, card_effect))

        elif isinstance(current_tile, SpecialTile):
            if current_tile.special_tile_type == SpecialTileType.GO:
                player.balance += 200
            elif current_tile.special_tile_type == SpecialTileType.GO_TO_JAIL:
                self.send_player_to_jail(player=player)
        else:
            self.logger.error(f"Failed to identify the type of tile for tile={current_tile}")


    def calculate_rent(self, property, dice_roll):
        if property.is_mortgaged == True:
            return 0
        if isinstance(property, Street):
            color_properties = [prop for prop in self.board.board if isinstance(prop, Street) and prop.color_set == property.color_set]
            owns_full_set = all(prop.owner == property.owner for prop in color_properties)
            if owns_full_set:
                if property.hotels > 0:
                    return property.rent["hotel"]
                elif property.houses > 0:
                    return property.rent.get(f"{property.houses}_house_rent", 0)
                else:
                    return property.rent["color_set"]
            else:
                return property.rent.get("no_color_set", 0)
        elif isinstance(property, Utility):
            utilities_owned = sum(1 for prop in property.owner.properties if isinstance(prop, Utility))
            return sum(dice_roll) * property.rent_multiplier[utilities_owned - 1]
        elif isinstance(property, Railroad):
            railroads_owned = sum(1 for prop in property.owner.properties if isinstance(prop, Railroad))
            return property.rent[railroads_owned - 1]
        return 0
            
    def get_streets_in_color_set(self, color_set_obj: ColorSet) -> List[Street]:
        """Helper to get all Street objects belonging to a given ColorSet."""
        return [
            tile for tile in self.board.board
            if isinstance(tile, Street) and tile.color_set == color_set_obj
        ]


    def player_can_build_on_property(self, player: Player, street: Street, building_type: BuildingType, check_even_build: bool = True) -> bool:
        if not isinstance(street, Street) or street.owner != player or street.is_mortgaged:
            return False
        if not self.player_has_complete_color_set(player, street.color_set):
            return False

        if building_type == BuildingType.HOUSE:
            if street.houses >= 4: # Max houses or has hotel
                return False
            if check_even_build:
                for other_s_in_set in self.get_streets_in_color_set(street.color_set):
                    if other_s_in_set.owner == player and other_s_in_set != street:
                        if other_s_in_set.hotels == 0 and other_s_in_set.houses < street.houses:
                            return False
            return True
        
        elif building_type == BuildingType.HOTEL:
            if street.houses != 4 or street.hotels >= 1:
                return False
            if check_even_build:
                for other_s_in_set in self.get_streets_in_color_set(street.color_set):
                    if other_s_in_set.owner == player and other_s_in_set != street:
                        if not (other_s_in_set.houses == 4 or other_s_in_set.hotels > 0):
                            return False
            return True
        return False

    def get_potential_building_auction_competitors(self, initiator: Player, building_type_to_auction: BuildingType, cost_basis_property: Street) -> List['Player']:
        competitors = []
        base_cost = cost_basis_property.color_set.house_cost if building_type_to_auction == BuildingType.HOUSE else cost_basis_property.color_set.hotel_cost

        for player in self.players:
            if player == initiator or player.is_bankrupt:
                continue

            if player.balance < base_cost:
                continue

            if self.player_can_build_type_on_any_property(player, building_type_to_auction):
                competitors.append(player)
                
        return competitors


    def player_can_build_type_on_any_property(self, player: Player, building_type: BuildingType) -> bool:
        for prop in player.properties:
            if isinstance(prop, Street):
                if self.player_can_build_on_property(player, prop, building_type, check_even_build=True):
                    return True
        return False
