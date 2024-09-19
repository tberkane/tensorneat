from typing import Callable

import jax
from jax import vmap, numpy as jnp
import numpy as np

from .species import SpeciesController
from .. import BaseAlgorithm
from tensorneat.common import State
from tensorneat.genome import BaseGenome


class NEAT(BaseAlgorithm):
    def __init__(
        self,
        genome: BaseGenome,
        pop_size: int,
        species_size: int = 10,
        max_stagnation: int = 15,
        species_elitism: int = 2,
        spawn_number_change_rate: float = 0.5,
        genome_elitism: int = 2,
        survival_threshold: float = 0.1,
        min_species_size: int = 1,
        compatibility_threshold: float = 2.0,
        species_fitness_func: Callable = jnp.max,
    ):
        self.genome = genome
        self.pop_size = pop_size
        self.species_controller = SpeciesController(
            pop_size,
            species_size,
            max_stagnation,
            species_elitism,
            spawn_number_change_rate,
            genome_elitism,
            survival_threshold,
            min_species_size,
            compatibility_threshold,
            species_fitness_func,
        )

    def setup(self, state=State()):
        # setup state
        state = self.genome.setup(state)

        k1, randkey = jax.random.split(state.randkey, 2)

        # initialize the population
        initialize_keys = jax.random.split(k1, self.pop_size)
        pop_nodes, pop_conns = vmap(self.genome.initialize, in_axes=(None, 0))(
            state, initialize_keys
        )

        # Initialize connection weights
        k2, randkey = jax.random.split(randkey)
        pop_conns = (
            jax.random.normal(k2, pop_conns.shape) * 0.1
        )  # Small initial weights

        state = state.register(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            generation=jnp.float32(0),
        )

        # initialize species state
        state = self.species_controller.setup(state, pop_nodes[0], pop_conns[0])

        return state.update(randkey=randkey)

    def ask(self, state, training_data=None, num_epochs=1):
        pop_nodes, pop_conns = state.pop_nodes, state.pop_conns

        if training_data is not None:
            for i in range(self.pop_size):
                print(f"Training network {i+1}/{self.pop_size}")
                nodes, conns = pop_nodes[i], pop_conns[i]
                for _ in range(num_epochs):
                    print(f"  Epoch {_+1}/{num_epochs}")
                    nodes, conns = self.train_network(
                        state, nodes, conns, training_data
                    )
                pop_nodes = pop_nodes.at[i].set(nodes)
                pop_conns = pop_conns.at[i].set(conns)

        print("Training completed")
        return pop_nodes, pop_conns

    def train_network(self, state, nodes, conns, training_data):
        print("  Computing gradients")
        # Compute gradients
        grads = self.genome.compute_gradients(
            state, nodes, conns, training_data["inputs"], training_data["targets"]
        )

        print("  Updating weights")
        # Update weights
        nodes, conns = self.update_weights(nodes, conns, grads)

        return nodes, conns

    def compute_loss(self, outputs, targets):
        # Compute cross-entropy loss
        epsilon = 1e-12  # Small value to avoid log(0)
        outputs = jnp.clip(
            outputs, epsilon, 1 - epsilon
        )  # Clip values to avoid numerical instability
        return -jnp.sum(targets * jnp.log(outputs)) / outputs.shape[0]

    def update_weights(self, nodes, conns, grads):
        learning_rate = 0.01  # You might want to make this configurable
        print(f"  Updating weights with learning rate: {learning_rate}")
        nodes = nodes - learning_rate * grads[0]
        conns = conns - learning_rate * grads[1]
        return nodes, conns

    def tell(self, state, fitness):
        state = state.update(generation=state.generation + 1)

        # tell fitness to species controller
        state, winner, loser, elite_mask = self.species_controller.update_species(
            state,
            fitness,
        )

        # create next population
        state = self._create_next_generation(state, winner, loser, elite_mask)

        # speciate the next population
        state = self.species_controller.speciate(state, self.genome.execute_distance)

        return state

    def transform(self, state, individual):
        nodes, conns = individual
        return self.genome.transform(state, nodes, conns)

    def forward(self, state, transformed, inputs):
        return self.genome.forward(state, transformed, inputs)

    @property
    def num_inputs(self):
        return self.genome.num_inputs

    @property
    def num_outputs(self):
        return self.genome.num_outputs

    def _create_next_generation(self, state, winner, loser, elite_mask):

        # find next node key for mutation
        all_nodes_keys = state.pop_nodes[:, :, 0]
        max_node_key = jnp.max(
            all_nodes_keys, where=~jnp.isnan(all_nodes_keys), initial=0
        )
        next_node_key = max_node_key + 1
        new_node_keys = jnp.arange(self.pop_size) + next_node_key

        # prepare random keys
        k1, k2, randkey = jax.random.split(state.randkey, 3)
        crossover_randkeys = jax.random.split(k1, self.pop_size)
        mutate_randkeys = jax.random.split(k2, self.pop_size)

        wpn, wpc = state.pop_nodes[winner], state.pop_conns[winner]
        lpn, lpc = state.pop_nodes[loser], state.pop_conns[loser]

        # batch crossover
        n_nodes, n_conns = vmap(
            self.genome.execute_crossover, in_axes=(None, 0, 0, 0, 0, 0)
        )(
            state, crossover_randkeys, wpn, wpc, lpn, lpc
        )  # new_nodes, new_conns

        # batch mutation
        m_n_nodes, m_n_conns = vmap(
            self.genome.execute_mutation, in_axes=(None, 0, 0, 0, 0)
        )(
            state, mutate_randkeys, n_nodes, n_conns, new_node_keys
        )  # mutated_new_nodes, mutated_new_conns

        # elitism don't mutate
        pop_nodes = jnp.where(elite_mask[:, None, None], n_nodes, m_n_nodes)
        pop_conns = jnp.where(elite_mask[:, None, None], n_conns, m_n_conns)

        return state.update(
            randkey=randkey,
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
        )

    def show_details(self, state, fitness):
        member_count = jax.device_get(state.species.member_count)
        species_sizes = [int(i) for i in member_count if i > 0]

        pop_nodes, pop_conns = jax.device_get([state.pop_nodes, state.pop_conns])
        nodes_cnt = (~np.isnan(pop_nodes[:, :, 0])).sum(axis=1)  # (P,)
        conns_cnt = (~np.isnan(pop_conns[:, :, 0])).sum(axis=1)  # (P,)

        max_node_cnt, min_node_cnt, mean_node_cnt = (
            max(nodes_cnt),
            min(nodes_cnt),
            np.mean(nodes_cnt),
        )

        max_conn_cnt, min_conn_cnt, mean_conn_cnt = (
            max(conns_cnt),
            min(conns_cnt),
            np.mean(conns_cnt),
        )

        print(
            f"\tnode counts: max: {max_node_cnt}, min: {min_node_cnt}, mean: {mean_node_cnt:.2f}\n",
            f"\tconn counts: max: {max_conn_cnt}, min: {min_conn_cnt}, mean: {mean_conn_cnt:.2f}\n",
            f"\tspecies: {len(species_sizes)}, {species_sizes}\n",
        )
