import logging
import numpy as np
from typing import List, Optional, Set, Tuple

from monopoly_gym.player import Player
from monopoly_gym.state import State
from monopoly_gym.action import (
    Action, ActionSpaceType, ActionManager, EndTurnAction, RollDiceAction, BuyAction,
    AuctionAction, MortgageAction, UnmortgageAction, BuildAction, SellBuildingAction,
    UseJailCardAction, PayJailFineAction, RollJailAction, BankruptcyAction,
    AuctionBidAction, AuctionFoldAction, RejectTradeAction
)
from monopoly_gym.tile import Property, Street, Railroad, Utility

logger = logging.getLogger(__name__)

class FixedPolicyFivePlayer(Player):
    def __init__(self, name: str, mgn_code: str, starting_balance: int = 1500):
        super().__init__(name, mgn_code, starting_balance)
        self.action_manager = ActionManager(ActionSpaceType.HIERARCHICAL)
        self.policy_name = "FixedPolicyFive (Cautious Landlord)"
        self.safety_buffer_general = 300
        self.max_build_houses = 1
        self.build_cash_threshold = 800

    def _get_property_obj_from_game_state(self, game_state: State, property_tile_index: int) -> Optional[Property]:
        if 0 <= property_tile_index < len(game_state.board.tiles):
            tile = game_state.board.tiles[property_tile_index]
            if isinstance(tile, Property):
                return tile
        return None

    def _get_owned_property_by_tile_index(self, tile_index: int) -> Optional[Property]:
        for prop in self.properties:
            if prop.index == tile_index:
                return prop
        return None
        
    def _get_all_streets_in_color_set(self, game_state: State, color_set_name: str) -> List[Street]:
        if not color_set_name: return []
        return [
            p for p in game_state.board.tiles
            if isinstance(p, Street) and p.color_set == color_set_name
        ]

    def _owns_full_color_set(self, game_state: State, street_obj: Street) -> bool:
        if not street_obj.color_set: return False
        all_props_in_set = self._get_all_streets_in_color_set(game_state, street_obj.color_set)
        if not all_props_in_set: return False
        for prop_in_set in all_props_in_set:
            if prop_in_set.owner != self:
                return False
        return True
    
    def _get_potential_mortgage_value(self, property_to_mortgage: Property, game_state: State) -> int:
        return property_to_mortgage.mortgage_value

    def decide_actions(self, game_state: State) -> List[Action]:
        player = self
        action_mask_full = self.action_manager.to_action_mask(game_state)
        action_types_mask = action_mask_full['action_type']
        am = self.action_manager

        def is_action_type_valid(action_name_str: str) -> bool:
            if action_name_str not in am.action_name_to_idx:
                return False
            return bool(action_types_mask[am.action_name_to_idx[action_name_str]])
        if game_state.auction_state and game_state.auction_state.player_to_act == player:
            prop_on_auction = game_state.auction_state.property
            min_bid = game_state.auction_state.current_bid + (game_state.auction_state.bid_increment or 1)
            max_bid_allowed = player.balance
            bid_limit = prop_on_auction.purchase_cost * 0.4 
            actual_bid = min(min_bid, int(bid_limit), max_bid_allowed)

            if actual_bid >= min_bid:
                 if is_action_type_valid('AuctionBidAction'):
                    return [AuctionBidAction(player=player, bid_amount=actual_bid)]
            
            if is_action_type_valid('AuctionFoldAction'):
                return [AuctionFoldAction(player=player)]
        if game_state.pending_trade and game_state.pending_trade.responder == player:
            if is_action_type_valid('RejectTradeAction'):
                return [RejectTradeAction(player=player)]
        if player.in_jail:
            if player.jail_free_cards > 0 and is_action_type_valid('UseJailCardAction'):
                return [UseJailCardAction(player=player)]
            if is_action_type_valid('RollJailAction'): # Prefer roll over pay
                return [RollJailAction(player=player)]
            if player.balance >= 50 + self.safety_buffer_general * 2 and is_action_type_valid('PayJailFineAction'): # Pay only if very rich
                return [PayJailFineAction(player=player)]
        if player.balance < 0:
            if is_action_type_valid('SellBuildingAction'):
                owned_streets_with_one_house = [
                    p for p in player.properties 
                    if isinstance(p, Street) and p.houses == 1 and p.hotels == 0
                ]
                if owned_streets_with_one_house:
                    owned_streets_with_one_house.sort(key=lambda s: s.purchase_cost, reverse=True)
                    return [SellBuildingAction(player=player, street=owned_streets_with_one_house[0], quantity=1)]

            if is_action_type_valid('MortgageAction'):
                unmortgaged_props = [p for p in player.properties if not p.is_mortgaged]
                unmortgaged_props.sort(key=lambda p_obj: p_obj.purchase_cost)
                for prop_to_mortgage in unmortgaged_props:
                    return [MortgageAction(player=player, property_obj=prop_to_mortgage)]
            
            if player.balance < 0 and is_action_type_valid('BankruptcyAction'):
                return [BankruptcyAction(player=player)]
        if is_action_type_valid('UnmortgageAction'):
            if player.balance > 1200: 
                mortgaged_props = [p for p in player.properties if p.is_mortgaged]
                mortgaged_props.sort(key=lambda p_obj: p_obj.unmortgage_cost) # Unmortgage cheapest
                for prop_to_unmortgage in mortgaged_props:
                    if player.balance >= prop_to_unmortgage.unmortgage_cost + 1000: # Huge buffer
                        return [UnmortgageAction(player=player, property_obj=prop_to_unmortgage)]
        if is_action_type_valid('BuildAction') and player.balance > self.build_cash_threshold:
            unmortgaged_owned_streets_no_buildings = []
            for prop in player.properties:
                if isinstance(prop, Street) and not prop.is_mortgaged and prop.houses == 0 and prop.hotels == 0:
                    if self._owns_full_color_set(game_state, prop):
                         unmortgaged_owned_streets_no_buildings.append(prop)
            unmortgaged_owned_streets_no_buildings.sort(key=lambda s: s.house_cost)

            for prop_to_build_on in unmortgaged_owned_streets_no_buildings:
                if player.balance >= prop_to_build_on.house_cost + self.safety_buffer_general:
                     return [BuildAction(player=player, street=prop_to_build_on, quantity=1)]
        current_tile_obj = self._get_property_obj_from_game_state(game_state, player.position)

        if is_action_type_valid('RollDiceAction'):
            return [RollDiceAction(player=player)]

        if isinstance(current_tile_obj, Property) and current_tile_obj.owner is None:
            if is_action_type_valid('BuyAction'):
                if player.balance >= current_tile_obj.purchase_cost + self.safety_buffer_general and current_tile_obj.purchase_cost <= 200:
                    return [BuyAction(player=player, property_obj=current_tile_obj)]
                elif player.balance < current_tile_obj.purchase_cost + self.safety_buffer_general and current_tile_obj.purchase_cost <= 200 and is_action_type_valid('MortgageAction'):
                    unmortgaged_owned_props = [p for p in player.properties if not p.is_mortgaged]
                    unmortgaged_owned_props.sort(key=lambda p_obj: p_obj.purchase_cost) # Mortgage cheapest owned
                    for prop_to_mortgage in unmortgaged_owned_props:
                        if player.balance + self._get_potential_mortgage_value(prop_to_mortgage, game_state) >= current_tile_obj.purchase_cost + self.safety_buffer_general / 2:
                            if current_tile_obj.purchase_cost < prop_to_mortgage.purchase_cost or len(player.properties) < 5:
                                return [MortgageAction(player=player, property_obj=prop_to_mortgage)]
            
            if is_action_type_valid('AuctionAction'): # If didn't buy
                return [AuctionAction(player=player, property_obj=current_tile_obj)]
        if is_action_type_valid('EndTurnAction'):
            return [EndTurnAction(player=player)]
        logger.warning(f"{self.policy_name} ({player.name}): Reached fallback. Action Mask: {action_types_mask}")
        if is_action_type_valid('EndTurnAction'): return [EndTurnAction(player=player)]
        if is_action_type_valid('BankruptcyAction') and player.balance < 0 : return [BankruptcyAction(player=player)]
        return []