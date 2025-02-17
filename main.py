import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random


# --------------------------
# Data Structures for Cards and Hands
# --------------------------
class Card:
    valid_ranks = "23456789TJQKA"
    valid_suits = "CDHS"  # Clubs, Diamonds, Hearts, Spades
    rank_names = {
        '2': "2", '3': "3", '4': "4", '5': "5", '6': "6",
        '7': "7", '8': "8", '9': "9", 'T': "10", 'J': "Jack",
        'Q': "Queen", 'K': "King", 'A': "Ace"
    }
    suit_names = {'C': "Clubs", 'D': "Diamonds", 'H': "Hearts", 'S': "Spades"}

    def __init__(self, card_str):
        card_str = card_str.strip().upper()
        if len(card_str) < 2:
            raise ValueError(f"Invalid card format: {card_str}")
        self.rank = card_str[0]
        self.suit = card_str[1]
        if self.rank not in Card.valid_ranks or self.suit not in Card.valid_suits:
            raise ValueError(f"Invalid card: {card_str}")

    def __repr__(self):
        return f"{self.rank}{self.suit}"


class PokerHand:
    def __init__(self, hole_cards, community_cards):
        # hole_cards and community_cards should be list of card strings.
        self.hole_cards = [Card(card) for card in hole_cards]
        self.community_cards = [Card(card) for card in community_cards]

    def all_cards(self):
        return self.hole_cards + self.community_cards

    def evaluate(self):
        return evaluate_poker_hand(self.all_cards())

    def __repr__(self):
        hole = ", ".join(str(card) for card in self.hole_cards)
        community = ", ".join(str(card) for card in self.community_cards)
        hand_rank = self.evaluate()
        return f"Hole Cards: [{hole}] | Community Cards: [{community}]  -->  {hand_rank}"


# --------------------------
# Hand Evaluation Function
# --------------------------
def evaluate_poker_hand(cards):
    """
    Evaluates a list of Card objects and returns a descriptive string
    such as "Ace High Card", "Pair of 10s", "Jack High Diamond Flush", etc.
    This is a simplified evaluator covering the main hand types.
    """
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    # Count ranks and suits
    rank_counts = {}
    suit_counts = {}
    for card in cards:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

    # Identify groups by count
    fours = [r for r, count in rank_counts.items() if count == 4]
    threes = [r for r, count in rank_counts.items() if count == 3]
    pairs = [r for r, count in rank_counts.items() if count == 2]

    # Check for flush: any suit with at least 5 cards
    flush_suit = None
    for suit, count in suit_counts.items():
        if count >= 5:
            flush_suit = suit
            break

    # Check for straight (using unique rank values)
    unique_vals = sorted({rank_values[card.rank] for card in cards}, reverse=True)
    # Add special case for wheel: Ace-2-3-4-5
    if 14 in unique_vals:
        unique_vals.append(1)
    straight_high = None
    for i in range(len(unique_vals) - 4):
        if unique_vals[i] - unique_vals[i + 4] == 4:
            straight_high = unique_vals[i]
            break

    # Check for straight flush if flush exists
    straight_flush_high = None
    if flush_suit:
        flush_cards = [card for card in cards if card.suit == flush_suit]
        flush_vals = sorted({rank_values[card.rank] for card in flush_cards}, reverse=True)
        if 14 in flush_vals:
            flush_vals.append(1)
        for i in range(len(flush_vals) - 4):
            if flush_vals[i] - flush_vals[i + 4] == 4:
                straight_flush_high = flush_vals[i]
                break

    # Use mapping for full names
    def rank_full(r):
        return Card.rank_names[r]

    suit_full = lambda s: Card.suit_names[s]

    # Determine best hand ranking (priority order)
    if straight_flush_high:
        high_name = [r for r, val in rank_values.items() if val == straight_flush_high][0]
        return f"Straight Flush, high card {rank_full(high_name)} of {suit_full(flush_suit)}"
    elif fours:
        four = max(fours, key=lambda r: rank_values[r])
        return f"Four of a Kind, {rank_full(four)}s"
    elif threes and (pairs or len(threes) > 1):
        three = max(threes, key=lambda r: rank_values[r])
        # If more than one three-of-a-kind, the second one acts as the pair.
        if len(threes) > 1:
            remaining = [r for r in threes if r != three] + pairs
            pair = max(remaining, key=lambda r: rank_values[r])
        else:
            pair = max(pairs, key=lambda r: rank_values[r])
        return f"Full House, {rank_full(three)}s over {rank_full(pair)}s"
    elif flush_suit:
        flush_cards = [card for card in cards if card.suit == flush_suit]
        high_flush = max(flush_cards, key=lambda card: rank_values[card.rank])
        return f"Flush, high card {rank_full(high_flush.rank)} of {suit_full(flush_suit)}"
    elif straight_high:
        high_name = [r for r, val in rank_values.items() if val == straight_high][0]
        return f"Straight, high card {rank_full(high_name)}"
    elif threes:
        three = max(threes, key=lambda r: rank_values[r])
        return f"Three of a Kind, {rank_full(three)}s"
    elif len(pairs) >= 2:
        sorted_pairs = sorted(pairs, key=lambda r: rank_values[r], reverse=True)
        return f"Two Pair, {rank_full(sorted_pairs[0])}s and {rank_full(sorted_pairs[1])}s"
    elif pairs:
        pair = max(pairs, key=lambda r: rank_values[r])
        return f"Pair of {rank_full(pair)}s"
    else:
        high_card = max(cards, key=lambda card: rank_values[card.rank])
        return f"High Card {rank_full(high_card.rank)} of {suit_full(high_card.suit)}"


# --------------------------
# 1. Opponent Range Estimation Model (Placeholder for later use)
# --------------------------
class OpponentRangeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(OpponentRangeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Using sigmoid to produce outputs between 0 and 1 (a dummy "range strength")
        x = torch.sigmoid(self.fc2(x))
        return x


# --------------------------
# Helper: Determine the current stage of the game
# --------------------------
def determine_stage(community_cards):
    """Determine the current stage of the game based on the number of community cards."""
    if len(community_cards) == 0:
        return "Pre-Flop"
    elif len(community_cards) == 3:
        return "Flop"
    elif len(community_cards) == 4:
        return "Turn"
    elif len(community_cards) == 5:
        return "River"
    else:
        return "Unknown Stage"


# --------------------------
# Helper: Infer player action from bet amounts
# --------------------------
def infer_actions(betting_order, bets):
    """
    Infers each player's action based on their bet.
    If a player bets -1, that's a fold.
    Otherwise, if the bet equals the current call amount, it is a check (if 0) or call.
    If the bet is higher than the current call amount, it's a raise (and updates the call amount).
    """
    current_call = 0.0
    actions = {}
    for pos in betting_order:
        bet = bets[pos]
        if bet == -1:
            actions[pos] = "fold"
        elif bet == current_call:
            if current_call == 0:
                actions[pos] = "check"
            else:
                actions[pos] = "call"
        elif bet > current_call:
            actions[pos] = "raise"
            current_call = bet
        else:
            actions[pos] = "call"
    return actions


# --------------------------
# Main Loop: Run rounds continuously
# --------------------------
def main():
    print("Welcome to the AI Poker Win Rate Estimator!")
    print("Note: Use -1 as the bet amount to indicate a fold.")
    print("Positions: Dealer is player 0, Small Blind is player 1, Big Blind is player 2.\n")

    # Game Setup: Get number of players and user position (0-indexed)
    num_players = int(input("Enter the number of players on the table: "))
    user_position = int(input("Enter your position (0-indexed): "))

    # Validate minimum number of players (at least 3 are needed for Texas Hold'em)
    if num_players < 3:
        print("Error: Texas Hold'em requires at least 3 players.")
        return

    # Get the user's hole cards once (memorized for subsequent rounds)
    user_cards_input = input("Enter your hole cards separated by a comma (e.g., AH, KD): ")
    user_cards = [card.strip().upper() for card in user_cards_input.split(",") if card.strip()]

    round_counter = 1
    # Initialize active players for the hand: all players start active
    active_players = list(range(num_players))

    while True:
        print("\n=== Round {} ===".format(round_counter))
        # If only one player remains, the hand is over.
        if len(active_players) == 1:
            print(f"Hand over! Only Player {active_players[0]} remains active.")
            # Option to start a new hand (reset active players)
            new_hand = input("Start a new hand? (y/n): ")
            if new_hand.lower() != 'y':
                print("Exiting the AI Poker Win Rate Estimator. Goodbye!")
                break
            else:
                active_players = list(range(num_players))
                round_counter = 1
                continue

        # Get community cards (if any) for the current round
        community_cards_input = input("Enter community cards separated by a comma (if none, leave blank): ")
        community_cards = [card.strip().upper() for card in community_cards_input.split(",") if card.strip()]

        # Build the player's current PokerHand using the memorized hole cards and current community cards.
        current_hand = PokerHand(user_cards, community_cards)

        # Determine current stage based on community cards
        stage = determine_stage(community_cards)

        # Determine betting order based on stage then filter to active players.
        if stage == "Pre-Flop":
            full_order = list(range(3, num_players)) + [0, 1, 2]
        else:
            full_order = list(range(1, num_players)) + [0]
        betting_order = [pos for pos in full_order if pos in active_players]

        # Get bets from players in the proper order (only for active players)
        bets = {}
        print("\n--- Enter Bets ---")
        for pos in betting_order:
            if pos == user_position:
                bet_input = input(f"Player {pos} (You) - Enter your bet amount: ")
            else:
                bet_input = input(f"Player {pos} - Enter bet amount: ")
            try:
                bet = float(bet_input)
            except:
                bet = 0.0
            bets[pos] = bet

        # Infer actions based on the bets and betting order
        actions = infer_actions(betting_order, bets)

        # Update active players: remove any player who just folded
        for pos in betting_order:
            if actions[pos] == "fold" and pos in active_players:
                active_players.remove(pos)

        # Compute the current total cumulated bet for this round (ignoring folds)
        cumulated_bet = sum(bet for bet in bets.values() if bet != -1)

        # Build players_info only for active players
        players_info = {}
        for pos in active_players:
            players_info[pos] = {
                "bet": bets.get(pos, "N/A"),
                "action": actions.get(pos, "N/A")
            }

        # Placeholder values for computed outputs
        placeholder_win_rate = 50.0  # Dummy win rate (50%)
        opponents_predictions = {}
        for pos in active_players:
            if pos != user_position:
                opponents_predictions[pos] = ["??", "??"]

        # Print Game Summary for the round
        print("\n--- Game Summary ---")
        print(f"Current Stage: {stage}")
        print(f"User Position: {user_position}")
        print(f"User Hole Cards: {user_cards}")
        print("Community Cards:", community_cards if community_cards else "None")
        print(f"Estimated Win Rate: {placeholder_win_rate:.2f}%")
        print(f"Total Cumulated Bet (Pot): {cumulated_bet}\n")
        print(f"Your Best Hand: {current_hand}\n")

        print("Active Players Info:")
        for pos in active_players:
            if pos == user_position:
                print(
                    f"  Position {pos} (You) - Bet: {players_info[pos]['bet']}, Action: {players_info[pos]['action']}")
            else:
                pred = opponents_predictions.get(pos, ["??", "??"])
                print(
                    f"  Position {pos} - Bet: {players_info[pos]['bet']}, Action: {players_info[pos]['action']}, Predicted Cards: {pred}")

        # Ask if the user wants to continue to the next betting round of the current hand.
        cont = input("\nContinue to next betting round? (y/n): ")
        if cont.lower() != 'y':
            print("Exiting the AI Poker Win Rate Estimator. Goodbye!")
            break

        round_counter += 1


if __name__ == "__main__":
    main()
