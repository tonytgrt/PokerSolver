# gto.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class OpponentCardPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1326):
        """
        Initializes the OpponentCardPredictor network.

        Args:
            input_size (int): Dimension of the input feature vector.
            hidden_size (int): Number of neurons in hidden layers.
            output_size (int): Number of output classes (fixed to 1326 for two-card combinations).
        """
        super(OpponentCardPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Precompute all unique two-card combinations in canonical order.
        self.all_two_card_combinations = self._compute_two_card_combinations()

    def _compute_two_card_combinations(self):
        """
        Computes and returns a list of all 1,326 unique two-card combinations.
        Cards are represented as strings in the format 'AH', 'TD', etc.
        """
        # Define all 52 cards using ranks and suits.
        ranks = "23456789TJQKA"
        suits = "CDHS"  # Clubs, Diamonds, Hearts, Spades
        cards = [r + s for r in ranks for s in suits]
        # Use itertools.combinations to get all unique two-card sets (order does not matter).
        combinations = list(itertools.combinations(cards, 2))
        return combinations  # List of tuples; length should be 1326

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Probability distribution over 1,326 classes for each sample.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        # Use softmax to convert logits to probabilities.
        probabilities = F.softmax(logits, dim=-1)
        return probabilities

    def predict_cards(self, features):
        """
        Given an input feature vector (or batch), predicts the opponent's hole card combination.

        Args:
            features (torch.Tensor): Tensor of shape (input_size,) or (batch_size, input_size).

        Returns:
            tuple or list of tuples: The predicted two-card combination(s) as tuples, e.g. ('AH', 'KD').
        """
        self.eval()
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)  # Convert to batch_size=1 if necessary.
            probs = self.forward(features)
            # For each sample, take the index with the maximum probability.
            _, indices = torch.max(probs, dim=-1)
            # Map each predicted index to its corresponding two-card combination.
            predicted_cards = [self.all_two_card_combinations[idx] for idx in indices.cpu().numpy()]
            if len(predicted_cards) == 1:
                return predicted_cards[0]
            return predicted_cards


# If run as main, perform a simple test.
if __name__ == "__main__":
    # Example usage:
    # Assume the input feature vector is of size 10 (you can adjust this as needed).
    input_size = 10
    model = OpponentCardPredictor(input_size=input_size, hidden_size=64, output_size=1326)

    # Create a dummy input (e.g., representing an opponent's betting features).
    dummy_input = torch.randn(1, input_size)

    # Get predicted probabilities over the 1326 starting hands.
    probabilities = model(dummy_input)
    print("Predicted probabilities shape:", probabilities.shape)

    # Get the most likely predicted opponent cards.
    predicted = model.predict_cards(dummy_input[0])
    print("Predicted opponent cards:", predicted)
