import unittest
from unittest.mock import MagicMock
from SimplifiedThreePL import SimplifiedThreePL  # Assuming the class is named SimplifiedThreePL
from Experiment import Experiment  # Assuming Experiment is the correct class

class TestSimplifiedThreePLInitialization(unittest.TestCase):

    def setUp(self):
        """Create a mock Experiment object for testing purposes."""
        self.experiment = MagicMock(spec=Experiment)
        self.experiment.conditions = []  # Initialize an empty list of conditions for simplicity

    def test_valid_initialization(self):
        """Test that constructor properly handles valid inputs."""
        try:
            # Initialize with a valid Experiment object
            model = SimplifiedThreePL(experiment=self.experiment)

            # Check that object is properly initialized
            self.assertIsInstance(model, SimplifiedThreePL)
            self.assertEqual(model.experiment, self.experiment)
            self.assertIsNone(model._base_rate)
            self.assertIsNone(model._discrimination)
            self.assertFalse(model._is_fitted)
        except Exception as e:
            self.fail(f"Initialization failed with exception: {e}")

    def test_access_parameter_before_fitting(self):
        """Test that constructor raises an exception if parameters are accessed before fitting."""
        try:
            model = SimplifiedThreePL(experiment=self.experiment)
            # Try to access the discrimination parameter before fitting
            model.get_discrimination()  # This should raise an exception
            self.fail("Expected ValueError for accessing discrimination before fitting, but no exception was raised.")
        except ValueError as e:
            self.assertEqual(str(e), "Model has not been fitted yet.")

        try:
            # Try to access the base rate parameter before fitting
            model.get_base_rate()  # This should raise an exception
            self.fail("Expected ValueError for accessing base rate before fitting, but no exception was raised.")
        except ValueError as e:
            self.assertEqual(str(e), "Model has not been fitted yet.")

    def test_invalid_input_mismatched_lengths(self):
        """Test that constructor raises an exception for invalid input (if necessary)."""
        # In this case, mismatched lengths is not relevant since we pass an Experiment object.
        # But we could mock an invalid condition list, for example:
        self.experiment.conditions = None  # This should trigger issues during model fitting
        try:
            model = SimplifiedThreePL(experiment=self.experiment)
            model.fit()  # This will fail due to invalid conditions
            self.fail("Expected ValueError for invalid input conditions, but no exception was raised.")
        except ValueError:
            pass  # Expected behavior

if __name__ == '__main__':
    unittest.main()
