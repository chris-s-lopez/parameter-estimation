import unittest
import numpy as np
from Experiment import Experiment
from SignalDetection import SignalDetection
from SimplifiedThreePL import SimplifiedThreePL

class TestSimplifiedThreePL(unittest.TestCase):
    
    def setUp(self):
        """Set up an Experiment object with predefined SignalDetection conditions."""
        self.experiment = Experiment()
        conditions = [
            SignalDetection(hits=8, misses=2, falseAlarms=1, correctRejections=9),
            SignalDetection(hits=7, misses=3, falseAlarms=2, correctRejections=8),
            SignalDetection(hits=6, misses=4, falseAlarms=3, correctRejections=7),
            SignalDetection(hits=5, misses=5, falseAlarms=4, correctRejections=6),
            SignalDetection(hits=4, misses=6, falseAlarms=5, correctRejections=5)
        ]
        
        for condition in conditions:
            self.experiment.add_condition(condition)
        
    def test_constructor_valid_input(self):
        """Test that constructor properly initializes with valid inputs."""
        model = SimplifiedThreePL(self.experiment)
        self.assertIsInstance(model, SimplifiedThreePL)
        
    def test_constructor_uninitialized_parameters(self):
        """Test that accessing parameters before fitting raises an error."""
        model = SimplifiedThreePL(self.experiment)
        with self.assertRaises(ValueError):
            model.get_discrimination()
        with self.assertRaises(ValueError):
            model.get_base_rate()

    def test_predict_output_range(self):
        """Test that predict() outputs values between 0 and 1."""
        model = SimplifiedThreePL(self.experiment)
        params = [1.0, 0.0]  # a=1.0, q=0.0
        probabilities = model.predict(params)
        for p in probabilities:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
    
    def test_predict_higher_base_rate(self):
        """Test that higher base rate values result in higher probabilities."""
        model = SimplifiedThreePL(self.experiment)
        params_low = [1.0, -2.0]  # a=1.0, q=-2.0 (low base rate)
        params_high = [1.0, 2.0]  # a=1.0, q=2.0 (high base rate)
        
        probs_low = model.predict(params_low)
        probs_high = model.predict(params_high)
        
        for low, high in zip(probs_low, probs_high):
            self.assertLess(low, high)
    
    def test_fit_parameter_stability(self):
        """Test that parameters remain stable when fitting multiple times."""
        model = SimplifiedThreePL(self.experiment)
        model.fit()
        a1, c1 = model.get_discrimination(), model.get_base_rate()
        model.fit()
        a2, c2 = model.get_discrimination(), model.get_base_rate()
        self.assertAlmostEqual(a1, a2, places=3)
        self.assertAlmostEqual(c1, c2, places=3)
    
    def test_integration_fixed_dataset(self):
        """Integration test with predefined accuracy rates."""
        exp = Experiment()
        accuracy_rates = [0.55, 0.60, 0.75, 0.90, 0.95]
        trials = 100
        
        for acc in accuracy_rates:
            hits = int(acc * trials)
            misses = trials - hits
            false_alarms = 1
            correct_rejections = 1
            exp.add_condition(SignalDetection(hits, misses, false_alarms, correct_rejections))
        
        model = SimplifiedThreePL(exp)
        model.fit()
        predictions = model.predict([model.get_discrimination(), np.log(model.get_base_rate() / (1 - model.get_base_rate()))])
        
        for pred, acc in zip(predictions, accuracy_rates):
            self.assertAlmostEqual(pred, acc, places=1)
    
    '''def test_corruption_invalid_parameter_assignment(self):
        #Test that users cannot manually overwrite fitted parameters.
        model = SimplifiedThreePL(self.experiment)
        model.fit()
        with self.assertRaises(AttributeError):
            model._discrimination = 10  # Should not allow direct modification
        with self.assertRaises(AttributeError):
            model._base_rate = -5  # Should not allow direct modification
        with self.asserRaises(AttributeError):
            model._logit_base_rate = 3.0'''

if __name__ == '__main__':
    unittest.main()
