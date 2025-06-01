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
from monopoly_gym.tile import Property, Street

logger = logging.getLogger(__name__)

class FixedPolicyTwoPlayer(Player):
    def __init__(self, name: str, mgn_code: str, starting_balance: int = 1500):
        super().__init__(name, mgn_code, starting_balance)
        self.action_manager = ActionManager(ActionSpaceType.HIERARCHICAL)
        self.policy_name = "FixedPolicyTwo (Aggressive Acquirer)"
        self.auction_bid_multiplier_completes_set = 1.2
        self.auction_bid_multiplier_helps_set = 1.0
        self.auction_bid_multiplier_other = 0.7
        self.min_cash_after_buy_via_mortgage = 50
        self.min_cash_after_build_via_mortgage = 0


    def _get_property_obj_from_game_state(self, game_state: State, property_tile_index: int) -> Optional[Property]:
        if 0 <= property_tile_index < len(game_state.board.tiles):
            tile = game_state.board.tiles[property_tile_index]
            if isinstance(tile, Property):
                return tile
        logger.warning(f"{self.policy_name}: Could not find property with tile index {property_tile_index}")
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

            bid_limit = 0
            is_street = isinstance(prop_on_auction, Street)
            
            if is_street and prop_on_auction.color_set:
                all_in_set = self._get_all_streets_in_color_set(game_state, prop_on_auction.color_set)
                owned_count = 0
                owned_others_count = 0
                for p_in_set in all_in_set:
                    if p_in_set.owner == player:
                        owned_count +=1
                        if p_in_set.index != prop_on_auction.index:
                             owned_others_count +=1
                
                if len(all_in_set) > 0 and owned_others_count == len(all_in_set) - 1: # Completes set
                    bid_limit = prop_on_auction.purchase_cost * self.auction_bid_multiplier_completes_set
                elif owned_count > 0: # Helps set (already owns some)
                    bid_limit = prop_on_auction.purchase_cost * self.auction_bid_multiplier_helps_set
                else: # Other street
                    bid_limit = prop_on_auction.purchase_cost * self.auction_bid_multiplier_other
            else: # Utility or Railroad
                bid_limit = prop_on_auction.purchase_cost * self.auction_bid_multiplier_other

            actual_bid = min(min_bid, int(bid_limit), max_bid_allowed)

            if actual_bid >= min_bid: # Ensure bid is at least min_bid and affordable
                 if is_action_type_valid('AuctionBidAction'):
                    logger.debug(f"{self.policy_name} ({player.name}): Bidding ${actual_bid} on {prop_on_auction.name}.")
                    return [AuctionBidAction(player=player, bid_amount=actual_bid)]
            
            if is_action_type_valid('AuctionFoldAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Folding auction for {prop_on_auction.name}.")
                return [AuctionFoldAction(player=player)]
        if game_state.pending_trade and game_state.pending_trade.responder == player:
            if is_action_type_valid('RejectTradeAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Declining trade from {game_state.pending_trade.proposer.name}.")
                return [RejectTradeAction(player=player)]
        if player.in_jail:
            if player.balance >= 50 and is_action_type_valid('PayJailFineAction'): # Pay if can afford fine
                logger.debug(f"{self.policy_name} ({player.name}): Paying jail fine.")
                return [PayJailFineAction(player=player)]
            if player.jail_free_cards > 0 and is_action_type_valid('UseJailCardAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Using Get Out of Jail Free card.")
                return [UseJailCardAction(player=player)]
            if is_action_type_valid('RollJailAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Rolling to get out of jail.")
                return [RollJailAction(player=player)]
        if player.balance < 0:
            logger.debug(f"{self.policy_name} ({player.name}): Negative balance (${player.balance}). Attempting to resolve.")
            if is_action_type_valid('SellBuildingAction'):
                owned_streets_with_buildings = [p for p in player.properties if isinstance(p, Street) and (p.houses > 0 or p.hotels > 0)]
                owned_streets_with_buildings.sort(key=lambda s: (s.hotels, s.houses, s.purchase_cost), reverse=True)
                for street_to_sell_from in owned_streets_with_buildings:
                    logger.debug(f"{self.policy_name} ({player.name}): Trying to sell building on {street_to_sell_from.name}.")
                    return [SellBuildingAction(player=player, street=street_to_sell_from, quantity=1)]
            
            if is_action_type_valid('MortgageAction'):
                unmortgaged_props = [p for p in player.properties if not p.is_mortgaged]
                props_to_consider_mortgaging = sorted(
                    unmortgaged_props,
                    key=lambda p_obj: (isinstance(p_obj, Street) and self._owns_full_color_set(game_state, p_obj), p_obj.purchase_cost)
                )
                for prop_to_mortgage in props_to_consider_mortgaging:
                    logger.debug(f"{self.policy_name} ({player.name}): Trying to mortgage {prop_to_mortgage.name} to cover debt.")
                    return [MortgageAction(player=player, property_obj=prop_to_mortgage)]
            
            if player.balance < 0 and is_action_type_valid('BankruptcyAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Declaring bankruptcy.")
                return [BankruptcyAction(player=player)]
        if is_action_type_valid('UnmortgageAction'):
            if player.balance > 1000 : # Significantly high cash
                mortgaged_owned_monopoly_props = []
                for prop in player.properties:
                    if prop.is_mortgaged and isinstance(prop, Street) and self._owns_full_color_set(game_state, prop):
                         mortgaged_owned_monopoly_props.append(prop)
                mortgaged_owned_monopoly_props.sort(key=lambda p_obj: -p_obj.purchase_cost)
                for prop_to_unmortgage in mortgaged_owned_monopoly_props:
                    if player.balance >= prop_to_unmortgage.unmortgage_cost + 800: # Still keep very high buffer
                        logger.debug(f"{self.policy_name} ({player.name}): Unmortgaging key monopoly property {prop_to_unmortgage.name}.")
                        return [UnmortgageAction(player=player, property_obj=prop_to_unmortgage)]
        if is_action_type_valid('BuildAction'):
            buildable_props_on_monopolies = [] # List of Street objects to build on
            
            distinct_owned_color_sets: Set[str] = set()
            for prop in player.properties:
                if isinstance(prop, Street) and prop.color_set and self._owns_full_color_set(game_state, prop):
                    all_in_set_unmortgaged = True
                    for s_in_set in self._get_all_streets_in_color_set(game_state, prop.color_set):
                        if s_in_set.is_mortgaged:
                            all_in_set_unmortgaged = False; break
                    if all_in_set_unmortgaged:
                        distinct_owned_color_sets.add(prop.color_set)
            sorted_color_sets = sorted(list(distinct_owned_color_sets), 
                                       key=lambda cs_name: -self._get_all_streets_in_color_set(game_state, cs_name)[0].purchase_cost)


            for cs_name in sorted_color_sets:
                streets_in_this_set = [p for p in player.properties if isinstance(p, Street) and p.color_set == cs_name]
                streets_in_this_set.sort(key=lambda s: (s.hotels, s.houses)) # Build evenly
                
                prop_to_build_on = streets_in_this_set[0]
                if prop_to_build_on.hotels == 1: continue 

                cost_to_build_one_house = prop_to_build_on.house_cost
                if player.balance >= cost_to_build_one_house:
                    logger.debug(f"{self.policy_name} ({player.name}): Building 1 house on {prop_to_build_on.name} (direct cash).")
                    return [BuildAction(player=player, street=prop_to_build_on, quantity=1)]
                elif is_action_type_valid('MortgageAction'):
                    props_to_mortgage_for_build = [p for p in player.properties if not p.is_mortgaged and not (isinstance(p, Street) and self._owns_full_color_set(game_state, p))]
                    props_to_mortgage_for_build.sort(key=lambda p_obj: p_obj.purchase_cost)
                    
                    if not props_to_mortgage_for_build:
                        props_to_mortgage_for_build = [p for p in player.properties if not p.is_mortgaged and p.color_set != cs_name]
                        props_to_mortgage_for_build.sort(key=lambda p_obj: p_obj.purchase_cost)


                    for prop_to_mortgage in props_to_mortgage_for_build:
                        if player.balance + self._get_potential_mortgage_value(prop_to_mortgage, game_state) >= cost_to_build_one_house + self.min_cash_after_build_via_mortgage:
                            logger.debug(f"{self.policy_name} ({player.name}): Planning to mortgage {prop_to_mortgage.name} to build on {prop_to_build_on.name}.")
                            return [MortgageAction(player=player, property_obj=prop_to_mortgage)]
        current_tile_obj = self._get_property_obj_from_game_state(game_state, player.position)

        if is_action_type_valid('RollDiceAction'):
            logger.debug(f"{self.policy_name} ({player.name}): Rolling dice.")
            return [RollDiceAction(player=player)]

        if isinstance(current_tile_obj, Property) and current_tile_obj.owner is None:
            if is_action_type_valid('BuyAction'):
                if player.balance >= current_tile_obj.purchase_cost:
                    logger.debug(f"{self.policy_name} ({player.name}): Buying {current_tile_obj.name} (direct cash).")
                    return [BuyAction(player=player, property_obj=current_tile_obj)]
                elif is_action_type_valid('MortgageAction'):
                    unmortgaged_non_monopoly_props = [
                        p for p in player.properties 
                        if not p.is_mortgaged and not (isinstance(p, Street) and self._owns_full_color_set(game_state, p))
                    ]
                    unmortgaged_non_monopoly_props.sort(key=lambda p_obj: p_obj.purchase_cost) 

                    for prop_to_mortgage in unmortgaged_non_monopoly_props:
                        if player.balance + self._get_potential_mortgage_value(prop_to_mortgage, game_state) >= current_tile_obj.purchase_cost + self.min_cash_after_buy_via_mortgage:
                            logger.debug(f"{self.policy_name} ({player.name}): Planning to mortgage {prop_to_mortgage.name} to buy {current_tile_obj.name}.")
                            return [MortgageAction(player=player, property_obj=prop_to_mortgage)]
            
            if is_action_type_valid('AuctionAction'):
                logger.debug(f"{self.policy_name} ({player.name}): Auctioning {current_tile_obj.name}.")
                return [AuctionAction(player=player, property_obj=current_tile_obj)]
        if is_action_type_valid('EndTurnAction'):
            logger.debug(f"{self.policy_name} ({player.name}): Ending turn.")
            return [EndTurnAction(player=player)]
        logger.warning(f"{self.policy_name} ({player.name}): Reached fallback, no specific action chosen. Action Mask: {action_types_mask}")
        valid_action_indices = np.where(action_types_mask)[0]
        if valid_action_indices.size > 0:
            if is_action_type_valid('EndTurnAction'): return [EndTurnAction(player=player)]
            if is_action_type_valid('BankruptcyAction') and player.balance < 0 : return [BankruptcyAction(player=player)]
            
            action_cls_idx = valid_action_indices[0]
            action_cls = am.action_classes[action_cls_idx]
            logger.error(f"{self.policy_name} ({player.name}): CRITICAL FALLBACK. Attempting to take first valid action type: {action_cls.__name__}. This may fail. Returning empty list.")
            return []
        
        logger.error(f"{self.policy_name} ({player.name}): No valid actions available at all according to mask.")
        return []