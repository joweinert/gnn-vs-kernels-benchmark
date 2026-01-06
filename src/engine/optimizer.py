from skopt import gp_minimize, dump
from skopt.utils import use_named_args
import time

class HyperParameterOptimizer:
    """
    Wrapper around scikit-optimize (gp_minimize) for Bayesian Optimization.
    """
    def __init__(self, objective_function, space, n_calls=20, random_state=42, output_path=None):
        """
        objective_function: function to minimize. Must accept arguments named matching dimensionality of 'space'.
                            Should return a scalar value (negative accuracy usually, since we minimize).
        space: List of Dimension objects (Real, Integer, Categorical)
        n_calls: Number of iterations for optimization
        """
        self.objective_function = objective_function
        self.space = space
        self.n_calls = n_calls
        self.random_state = random_state
        self.output_path = output_path
        
    def run(self):
        print(f"Starting optimization with {self.n_calls} calls...")
        start_time = time.perf_counter()
        
        @use_named_args(self.space)
        def decorated_objective(**params):
            return self.objective_function(**params)
            
        res = gp_minimize(
            decorated_objective,
            self.space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            verbose=True
        )
        
        end_time = time.perf_counter()
        print(f"Optimization finished in {end_time - start_time:.2f} seconds.")
        print(f"Best parameters: {res.x}")
        print(f"Best score: {res.fun}")
        
        # aand the gp min result pkl
        if self.output_path:
            dump(res, self.output_path, store_objective=False)
            
        return res
