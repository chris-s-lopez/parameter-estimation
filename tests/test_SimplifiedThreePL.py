import unittest
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):

    def setUp(self):
        """Set up experiment and conditions."""
        # Create a sample SignalDetection object
        self.sdt1 = SignalDetection(hits=50, misses=10, falseAlarms=5, correctRejections=35)
        self.sdt2 = SignalDetection(hits=60, misses=5, falseAlarms=3, correctRejections=42)

        # Create an Experiment and add conditions
        self.experiment = Experiment()
        self.experiment.add_condition(self.sdt1, label="Condition 1")
        self.experiment.add_condition(self.sdt2, label="Condition 2")

        # Create the SimplifiedThreePL model
        self.model = SimplifiedThreePL(self.experiment)

    def test_initialization(self):
        """Test that the initialization of the SimplifiedThreePL object works properly."""
        self.assertIsInstance(self.model, SimplifiedThreePL)

    def test_summary(self):
        """Test that the summary method works and returns correct data."""
        summary = self.model.summary()
        self.assertEqual(summary['n_total'], 2)
        self.assertEqual(summary['n_conditions'], 2)

    def test_predict_method(self):
        """Test that the predict method returns values between 0 and 1."""
        parameters = {'discrimination': 1.0, 'base_rate': 0.5}
        predictions = self.model.predict(parameters)
        for prob in predictions:
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)

    def test_negative_log_likelihood(self):
        """Test that the negative log likelihood function computes correctly."""
        parameters = {'discrimination': 1.0, 'base_rate': 0.5}
        nll = self.model.negative_log_likelihood(parameters)
        self.assertIsInstance(nll, float)

if __name__ == '__main__':
    unittest.main()
