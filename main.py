# begin main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from gto import OpponentCardPredictor  # Import the model from gto.py
import itertools

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
        # hole_cards and community_cards should be lists of card strings.
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
# Convert Hand Description into a Numeric Rank (for Simulation Comparison)
# --------------------------
def hand_rank_value(hand_description):
    hand_order = {
        "Straight Flush": 9,
        "Four of a Kind": 8,
        "Full House": 7,
        "Flush": 6,
        "Straight": 5,
        "Three of a Kind": 4,
        "Two Pair": 3,
        "Pair": 2,
        "High Card": 1
    }
    for key, val in hand_order.items():
        if hand_description.startswith(key):
            return val
    return 0

# --------------------------
# (Old Dummy) Function to Predict Opponent's Cards
# (Now replaced by the neural network from gto.py)
# --------------------------
# def predict_opponent_cards(action):
#     if action == "raise":
#         return ["AH", "KH"]
#     elif action == "call":
#         return ["QD", "JC"]
#     elif action == "check":
#         return ["9S", "8S"]
#     else:
#         return []

# --------------------------
# Monte Carlo Simulation to Estimate Win Rate
# --------------------------
def monte_carlo_win_rate(user_hole_cards, community_cards, num_opponents, num_simulations=500):
    wins = 0
    ties = 0
    # Create a full deck of card strings
    deck = [r+s for r in Card.valid_ranks for s in Card.valid_suits]
    for _ in range(num_simulations):
        # Remove known cards from the deck
        known_cards = set(user_hole_cards + community_cards)
        remaining_deck = [card for card in deck if card not in known_cards]
        sim_community = community_cards.copy()
        # Complete community cards if necessary (simulate board runout)
        while len(sim_community) < 5:
            card = random.choice(remaining_deck)
            sim_community.append(card)
            remaining_deck.remove(card)
        # Evaluate user's hand
        user_hand = PokerHand(user_hole_cards, sim_community)
        user_strength = hand_rank_value(user_hand.evaluate())
        opponent_strengths = []
        # Copy remaining deck for opponent sampling
        rem_deck_copy = remaining_deck.copy()
        for _ in range(num_opponents):
            opp_hand_cards = random.sample(rem_deck_copy, 2)
            for card in opp_hand_cards:
                rem_deck_copy.remove(card)
            opp_hand = PokerHand(opp_hand_cards, sim_community)
            opp_strength = hand_rank_value(opp_hand.evaluate())
            opponent_strengths.append(opp_strength)
        max_opp_strength = max(opponent_strengths) if opponent_strengths else 0
        if user_strength > max_opp_strength:
            wins += 1
        elif user_strength == max_opp_strength:
            ties += 1
    win_rate = (wins + 0.5 * ties) / num_simulations
    return win_rate

# --------------------------
# Helper: Determine the Current Stage of the Game
# --------------------------
def determine_stage(community_cards):
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
# Helper: Infer Player Action from Bet Amounts
# --------------------------
def infer_actions(betting_order, bets):
    current_call = 0.0
    actions = {}
    for pos in betting_order:
        bet = bets[pos]
        if bet == -1:
            actions[pos] = "fold"
        elif bet == current_call:
            actions[pos] = "check" if current_call == 0 else "call"
        elif bet > current_call:
            actions[pos] = "raise"
            current_call = bet
        else:
            actions[pos] = "call"
    return actions

# --------------------------
# Helper: Encode Opponent Features for GTO Predictor
# --------------------------
def encode_features(action, bet):
    """
    Encodes an opponent's betting behavior into a fixed-length feature vector.
    Here we use a simple encoding:
      - Action mapping: "raise" -> 1.0, "call" -> 0.5, "check" -> 0.0.
      - Bet normalized (assume bet is in dollars; divide by 100).
      - The remaining dimensions are padded with zeros to form a vector of size 10.
    """
    mapping = {"raise": 1.0, "call": 0.5, "check": 0.0}
    action_val = mapping.get(action, 0.0)
    bet_norm = bet / 100.0 if isinstance(bet, (int, float)) else 0.0
    features = torch.zeros(10)
    features[0] = action_val
    features[1] = bet_norm
    return features

# --------------------------
# Main Loop: Run Rounds Continuously
# --------------------------
def main():
    print("Welcome to the AI Poker Win Rate Estimator!")
    print("Note: Use -1 as the bet amount to indicate a fold.")
    print("Positions: Dealer is player 0, Small Blind is player 1, Big Blind is player 2.\n")

    num_players = int(input("Enter the number of players on the table: "))
    user_position = int(input("Enter your position (0-indexed): "))

    if num_players < 3:
        print("Error: Texas Hold'em requires at least 3 players.")
        return

    user_cards_input = input("Enter your hole cards separated by a comma (e.g., AH, KD): ")
    user_cards = [card.strip().upper() for card in user_cards_input.split(",") if card.strip()]

    # Instantiate the opponent card predictor (from gto.py)
    predictor = OpponentCardPredictor(input_size=10, hidden_size=64, output_size=1326)
    predictor.eval()  # Set to evaluation mode (assumes pretrained weights if available)

    round_counter = 1
    active_players = list(range(num_players))

    while True:
        print(f"\n=== Round {round_counter} ===")
        if len(active_players) == 1:
            print(f"Hand over! Only Player {active_players[0]} remains active.")
            new_hand = input("Start a new hand? (y/n): ")
            if new_hand.lower() != 'y':
                print("Exiting the AI Poker Win Rate Estimator. Goodbye!")
                break
            else:
                active_players = list(range(num_players))
                round_counter = 1
                continue

        community_cards_input = input("Enter community cards separated by a comma (if none, leave blank): ")
        community_cards = [card.strip().upper() for card in community_cards_input.split(",") if card.strip()]

        current_hand = PokerHand(user_cards, community_cards)
        stage = determine_stage(community_cards)

        if stage == "Pre-Flop":
            full_order = list(range(3, num_players)) + [0, 1, 2]
        else:
            full_order = list(range(1, num_players)) + [0]
        betting_order = [pos for pos in full_order if pos in active_players]

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

        actions = infer_actions(betting_order, bets)
        for pos in betting_order:
            if actions[pos] == "fold" and pos in active_players:
                active_players.remove(pos)

        cumulated_bet = sum(bet for bet in bets.values() if bet != -1)
        players_info = {pos: {"bet": bets.get(pos, "N/A"), "action": actions.get(pos, "N/A")}
                        for pos in active_players}

        # Use the GTO predictor to predict opponents' hole cards
        opponents_predictions = {}
        for pos in active_players:
            if pos != user_position:
                # Encode features from the opponent's action and bet
                feat = encode_features(actions.get(pos, "N/A"), bets.get(pos, 0.0))
                predicted_cards = predictor.predict_cards(feat)
                opponents_predictions[pos] = predicted_cards

        num_opponents = len(active_players) - 1
        win_rate = monte_carlo_win_rate(user_cards, community_cards, num_opponents, num_simulations=500)

        print("\n--- Game Summary ---")
        print(f"Current Stage: {stage}")
        print(f"User Position: {user_position}")
        print(f"User Hole Cards: {user_cards}")
        print("Community Cards:", community_cards if community_cards else "None")
        print(f"Estimated Win Rate: {win_rate*100:.2f}%")
        print(f"Total Cumulated Bet (Pot): {cumulated_bet}\n")
        print(f"Your Best Hand: {current_hand}\n")

        print("Active Players Info:")
        for pos in active_players:
            if pos == user_position:
                print(f"  Position {pos} (You) - Bet: {players_info[pos]['bet']}, Action: {players_info[pos]['action']}")
            else:
                pred = opponents_predictions.get(pos, ("??", "??"))
                print(f"  Position {pos} - Bet: {players_info[pos]['bet']}, Action: {players_info[pos]['action']}, Predicted Cards: {pred}")

        cont = input("\nContinue to next betting round? (y/n): ")
        if cont.lower() != 'y':
            print("Exiting the AI Poker Win Rate Estimator. Goodbye!")
            break

        round_counter += 1

if __name__ == "__main__":
    main()
# end main.py
