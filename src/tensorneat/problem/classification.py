from jax import jit
import jax.numpy as jnp

from tensorneat.problem.base import BaseProblem
from tensorneat.common import State
from typing import Callable


class ClassificationProblem(BaseProblem):
    jitable = True

    def __init__(self, test_data):
        self.test_data = test_data
        self._input_shape = self.test_data["inputs"].shape[1:]
        self._output_shape = self.test_data["targets"].shape[1:]

    @jit
    def evaluate(self, state: State, randkey, act_func: Callable, params):
        outputs = act_func(state, params, self.test_data["inputs"])
        accuracy = jnp.mean(
            jnp.argmax(outputs, axis=1) == jnp.argmax(self.test_data["targets"], axis=1)
        )
        return accuracy

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    def show(self, state: State, randkey, act_func: Callable, params, num_samples=10):
        """
        Show predictions for a few samples from the test set
        """
        # Select a few random samples
        indices = jnp.random.choice(
            randkey,
            self.test_data["inputs"].shape[0],
            shape=(num_samples,),
            replace=False,
        )
        inputs = self.test_data["inputs"][indices]
        true_targets = self.test_data["targets"][indices]

        # Get predictions
        outputs = act_func(state, params, inputs)
        predicted_classes = jnp.argmax(outputs, axis=1)
        true_classes = jnp.argmax(true_targets, axis=1)

        # Print results
        print("\nSample predictions:")
        print("Index | Input | Predicted | True")
        print("-" * 40)
        for i in range(num_samples):
            print(
                f"{indices[i]:5d} | {inputs[i]} | {predicted_classes[i]:9d} | {true_classes[i]:4d}"
            )

        # Calculate and print overall accuracy
        accuracy = self.evaluate(state, randkey, act_func, params)
        print(f"\nOverall accuracy: {accuracy:.4f}")

    def setup(self, state: State = State()):
        # If any setup is needed, do it here
        # For this problem, we don't need any additional setup
        return state
