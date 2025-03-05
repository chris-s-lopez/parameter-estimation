from scipy.optimize import minimize
import numpy as np

class SimplifiedThreePL:
    def __init__(self, experiment):
        """Initialize the SimplifiedThreePL model with an Experiment object."""
        self.experiment = experiment  # The Experiment object containing SignalDetection conditions
        self._base_rate = None  # Base rate parameter (c)
        self._discrimination = None  # Discrimination parameter (a)
        self._logit_base_rate = None  # Logit of base rate parameter
        self._is_fitted = False  # Indicates if the model is fitted

    def summary(self):
        """Returns a dictionary summarizing the experiment data."""
        n_total = 0
        n_correct = 0
        n_incorrect = 0
        n_conditions = len(self.experiment.conditions)  # Get number of conditions

        # Iterate over the conditions in the experiment
        for condition in self.experiment.conditions:
            n_total += condition.n_total_responses()  # Add total responses for the condition
            n_correct += condition.n_correct_responses()  # Add correct responses for the condition
            n_incorrect += condition.n_incorrect_responses()  # Add incorrect responses for the condition

        # Return the summary dictionary
        return {
            'n_total': n_total,
            'n_correct': n_correct,
            'n_incorrect': n_incorrect,
            'n_conditions': n_conditions
        }

    def predict(self, parameters):
        """Returns the probability of a correct response for each condition, given the parameters."""
        a, c = parameters  # Unpack the parameters (a = discrimination, c = base_rate)

        # Predict the probability of a correct response in each condition
        probabilities = []
        for condition in self.experiment.conditions:
            hit_rate = condition.hit_rate()  # Get the hit rate for each condition
            false_alarm_rate = condition.false_alarm_rate()  # Get the false alarm rate

            # Compute the probability using the 3PL model formula
            prob_correct = c + (1 - c) / (1 + np.exp(-a * (hit_rate - false_alarm_rate)))
            probabilities.append(prob_correct)
        
        return np.array(probabilities)

    def negative_log_likelihood(self, parameters):
        """Computes the negative log-likelihood of the data given the parameters."""
        a, c = parameters  # Unpack the parameters (a = discrimination, c = base_rate)
        nll = 0  # Initialize the negative log-likelihood

        # For each condition in the experiment, compute the negative log-likelihood
        for condition in self.experiment.conditions:
            hit_rate = condition.hit_rate()
            false_alarm_rate = condition.false_alarm_rate()

            # Calculate the probability of correct response for the current condition
            prob_correct = self.predict(parameters)[self.experiment.conditions.index(condition)]
            
            # Log-likelihood calculation (assuming binomial distribution)
            n_correct = condition.n_correct_responses()
            n_incorrect = condition.n_incorrect_responses()
            
            # Calculate the negative log-likelihood for the current condition
            nll += -n_correct * np.log(prob_correct) - n_incorrect * np.log(1 - prob_correct)

        return nll

    def fit(self):
        """Fit the model using maximum likelihood estimation (MLE) to find the best-fitting parameters."""
        # Initial guess for the discrimination and base rate parameters
        initial_guess = [1.0, 0.5]  # initial guess for a and c
        
        # Minimize the negative log-likelihood
        result = minimize(self.negative_log_likelihood, initial_guess, bounds=[(0, None), (0, 1)])

        # If optimization is successful, set the parameters
        if result.success:
            self._discrimination = result.x[0]
            self._base_rate = result.x[1]
            self._logit_base_rate = np.log(self._base_rate / (1 - self._base_rate))
            self._is_fitted = True
        else:
            raise ValueError("Model fitting failed. Optimization did not converge.")
