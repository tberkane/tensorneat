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


@jit
def forward(params, x, num_inputs, num_outputs, num_hidden, total_nodes, connections):
    batch_size = x.shape[0]

    # Use jnp.zeros_like with a known shape for initialization
    nodes = jnp.zeros_like(x, shape=(batch_size, total_nodes))
    nodes = nodes.at[:, :num_inputs].set(x)

    # Pre-compute the incoming connections for each node
    incoming_connections = {i: [] for i in range(total_nodes)}
    for j, k in connections:
        if j >= total_nodes or k >= total_nodes:
            print(f"Warning: Invalid connection {j} -> {k}. Total nodes: {total_nodes}")
            continue
        incoming_connections[k].append(j)

    for i in range(num_inputs, total_nodes):
        incoming = incoming_connections[i]
        node_value = jnp.sum(
            jnp.array([nodes[:, j] * params[f"w_{j}_{i}"][0] for j in incoming]),
            axis=0,
        )
        nodes = nodes.at[:, i].set(jax.nn.tanh(node_value))

    return jax.nn.softmax(nodes[:, -num_outputs:])


forward = jit(forward, static_argnums=(2, 3, 4, 5))


@jit
def loss(params, x, y, num_inputs, num_outputs, num_hidden, total_nodes, connections):
    preds = forward(
        params, x, num_inputs, num_outputs, num_hidden, total_nodes, connections
    )
    return -jnp.mean(jnp.sum(y * jnp.log(preds + 1e-8), axis=-1))


loss = jit(loss, static_argnums=(3, 4, 5, 6))


def train_network(
    num_inputs,
    num_outputs,
    num_hidden,
    connections,
    data,
    num_epochs,
    learning_rate=0.01,
):
    # Initialize the network
    key = random.PRNGKey(0)
    params = init_params(num_inputs, num_outputs, num_hidden, connections, key)

    # Prepare data
    x = jnp.array(data["inputs"])
    y = jax.nn.one_hot(jnp.array(data["targets"]), num_outputs)

    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    total_nodes = num_inputs + num_hidden + num_outputs

    # Training loop
    for epoch in range(num_epochs):
        loss_value, grads = jax.value_and_grad(loss)(
            params,
            x,
            y,
            num_inputs,
            num_outputs,
            num_hidden,
            total_nodes,
            connections,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value:.4f}")

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
                total_nodes = (
                    jnp.max(jnp.maximum(conns[:, 0], conns[:, 1])).astype(int) + 1
                )
                num_hidden = total_nodes - num_inputs - num_outputs
                connections = [
                    (conns[k, 0].astype(int), conns[k, 1].astype(int))
                    for k in range(conns.shape[0])
                ]
                print_network_info(num_inputs, num_outputs, num_hidden, connections)

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


# Add this debugging function
def print_network_info(num_inputs, num_outputs, num_hidden, connections):
    total_nodes = num_inputs + num_hidden + num_outputs
    print(f"Network info:")
    print(f"  Inputs: {num_inputs}")
    print(f"  Outputs: {num_outputs}")
    print(f"  Hidden: {num_hidden}")
    print(f"  Total nodes: {total_nodes}")


# In your NEAT class, before calling train_network:
