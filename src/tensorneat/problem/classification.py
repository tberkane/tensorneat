from jax import jit

from tensorneat.problem.base import BaseProblem
import jax.numpy as jnp


class ClassificationProblem(BaseProblem):
    def __init__(self, test_data):
        self.test_data = test_data

    @jit
    def evaluate(self, state, key, forward, transformed):
        outputs = forward(state, transformed, self.test_data["inputs"])
        accuracy = jnp.mean(
            jnp.argmax(outputs, axis=1) == jnp.argmax(self.test_data["targets"], axis=1)
        )
        return accuracy
