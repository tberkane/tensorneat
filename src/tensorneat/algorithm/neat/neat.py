from typing import Callable

import jax
from jax import vmap, numpy as jnp
import numpy as np

from .species import SpeciesController
from .. import BaseAlgorithm
from tensorneat.common import State
from tensorneat.genome import BaseGenome
from jax import grad, jit, random
import optax


def init_params(num_inputs, num_outputs, num_hidden, connections, key):
    total_nodes = num_inputs + num_hidden + num_outputs
    params = {}
    for i, j in connections:
        k = random.split(key)[0]
        params[f"w_{i}_{j}"] = random.normal(k, (1,)) * jnp.sqrt(2.0 / total_nodes)
    return params


def forward(params, x, num_inputs, num_outputs, num_hidden, connections):
    total_nodes = num_inputs + num_hidden + num_outputs
    nodes = jnp.zeros(total_nodes)
    nodes = nodes.at[:num_inputs].set(x)
    for i in range(num_inputs, total_nodes):
        incoming = [j for j, k in connections if k == i]
        if incoming:
            node_value = jnp.sum(
                jnp.array([nodes[j] * params[f"w_{j}_{i}"][0] for j in incoming])
            )
            nodes = nodes.at[i].set(jax.nn.tanh(node_value))
    return jax.nn.softmax(nodes[-num_outputs:])


def loss(params, x, y, num_inputs, num_outputs, num_hidden, connections):
    preds = forward(params, x, num_inputs, num_outputs, num_hidden, connections)
    return -jnp.mean(jnp.sum(y * jnp.log(preds + 1e-8), axis=-1))


def train_network(
    num_inputs,
    num_outputs,
    num_hidden,
    connections,
    data,
    num_epochs,
    batch_size=32,
    learning_rate=0.01,
):
    # Initialize the network
    key = random.PRNGKey(0)
    params = init_params(num_inputs, num_outputs, num_hidden, connections, key)

    # Prepare data
    x = jnp.array(data["inputs"])
    y = jax.nn.one_hot(jnp.array(data["targets"]), num_outputs)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Training loop
    num_batches = len(x) // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(num_batches):
            batch_x = x[i * batch_size : (i + 1) * batch_size]
            batch_y = y[i * batch_size : (i + 1) * batch_size]
            loss_value, grads = jax.value_and_grad(loss)(
                params,
                batch_x,
                batch_y,
                num_inputs,
                num_outputs,
                num_hidden,
                connections,
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss_value

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / num_batches:.4f}")

    # Prepare the output weights
    trained_weights = {key: float(value[0]) for key, value in params.items()}

    return trained_weights


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
                nodes = pop_nodes[i]
                conns = pop_conns[i]

                num_inputs = self.num_inputs
                num_outputs = self.num_outputs
                num_hidden = nodes.shape[1] - num_inputs - num_outputs
                connections = [
                    (conns[k, 0].astype(int), conns[k, 1].astype(int))
                    for k in range(conns.shape[0])
                ]

                # Train the network
                trained_weights = train_network(
                    num_inputs,
                    num_outputs,
                    num_hidden,
                    connections,
                    training_data,
                    num_epochs,
                )

                # Convert trained weights back to the right format
                for j, (start, end) in enumerate(connections):
                    weight_key = f"w_{start}_{end}"
                    if weight_key in trained_weights:
                        nodes = nodes.at[end, start].set(
                            jnp.array(trained_weights[weight_key])
                        )

                pop_nodes = pop_nodes.at[i].set(nodes)
                # No need to update pop_conns as connections haven't changed

        print("Training completed")
        return pop_nodes, pop_conns

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
