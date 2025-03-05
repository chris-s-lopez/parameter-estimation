import numpy as np
from scipy.optimize import minimize
from Experiment import Experiment

class SimplifiedThreePL:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self._logit_base_rate = None  # q parameter
        self._base_rate = None  # c parameter
        self._discrimination = None  # a parameter
        self._is_fitted = False

    def summary(self):
        """Returns a summary of the experiment data."""
        n_total = sum(sdt.n_total_responses() for sdt in self.experiment.conditions)
        n_correct = sum(sdt.n_correct_responses() for sdt in self.experiment.conditions)
        n_incorrect = sum(sdt.n_incorrect_responses() for sdt in self.experiment.conditions)
        n_conditions = len(self.experiment.conditions)
        
        return {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_conditions": n_conditions
        }

    def predict(self, parameters):
        """Returns probability of correct response in each condition given parameters."""
        a, q = parameters
        c = 1 / (1 + np.exp(-q))  # Inverse logit function
        b = [2, 1, 0, -1, -2]  # Fixed difficulty parameters
        
        theta = 0  # Fixed theta
        probabilities = [c + (1 - c) / (1 + np.exp(-a * (theta - bi))) for bi in b]
        
        return probabilities

    def negative_log_likelihood(self, parameters):
        """Computes negative log-likelihood of the data given the parameters."""
        probabilities = self.predict(parameters)
        log_likelihood = 0
        
        for i, sdt in enumerate(self.experiment.conditions):
            nic = sdt.n_correct_responses()
            nie = sdt.n_incorrect_responses()
            p = probabilities[i]
            log_likelihood += nic * np.log(p) + nie * np.log(1 - p)
        
        return -log_likelihood  # Negative log-likelihood for minimization

    def fit(self):
        """Finds the best-fitting discrimination parameter and base rate parameter."""
        result = minimize(self.negative_log_likelihood, x0=[1, 0], method='L-BFGS-B')
        
        if result.success:
            self._discrimination, self._logit_base_rate = result.x
            self._base_rate = 1 / (1 + np.exp(-self._logit_base_rate))  # Convert q to c
            self._is_fitted = True
        else:
            raise RuntimeError("Optimization failed")

    def get_discrimination(self):
        """Returns the estimate of the discrimination parameter a."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        """Returns the estimate of the base rate parameter c."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self._base_rate
