from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome
from tensorneat.problem.func_fit import XOR3d
from tensorneat.common import Act

if __name__ == "__main__":
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=10000,
            species_size=20,
            survival_threshold=0.01,
            genome=DefaultGenome(
                num_inputs=3,
                num_outputs=1,
                init_hidden_layers=(),
                output_transform=Act.standard_sigmoid,
            ),
        ),
        problem=XOR3d(),
        generation_limit=500,
        fitness_target=-1e-6,  # float32 precision
        seed=42,
    )

    # initialize state
    state = pipeline.setup()
    # run until terminate
    state, best = pipeline.auto_run(state)
    # show result
    pipeline.show(state, best)
