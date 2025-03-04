# SimplifiedThreePL.py
class SimplifiedThreePL:
    def __init__(self, difficulty, discrimination, guessing):
        """Initialize the parameters of the 3PL model."""
        self.difficulty = difficulty
        self.discrimination = discrimination
        self.guessing = guessing

    def probability(self, theta):
        """
        Compute the probability of a correct response.
        Arguments:
            theta: The ability parameter.
        Returns:
            probability of a correct response according to the 3PL model.
        """
        exp_term = 1 + (self.discrimination * (theta - self.difficulty))
        return self.guessing + (1 - self.guessing) / exp_term
