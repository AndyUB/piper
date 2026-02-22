import dataclasses
from typing import Dict

import ray
import torch
from torch.nn.utils import parameters_to_vector

from src.piper_compile import piper_setup
from src.piper_coordinator import PiperProgramCoordinator
from src.piper_exec import piper_exec
from src.piper_utils import piper_metadata
from test.models.llama import LLAMA_DEBUG, Transformer
from test.schedule_helpers import pp2_interleaved_1f1b_grid_schedule_mb1


SEED = 1337


@dataclasses.dataclass
class TestArgs:
    dp: int = 1
    pp: int = 2
    seq_len: int = 16


def _flatten_stage_param_vectors(stage_to_vec: Dict[int, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [stage_to_vec[stage_id] for stage_id in sorted(stage_to_vec.keys())]
    )


def run_baseline_step(config, seq_len: int, x: torch.Tensor, y: torch.Tensor) -> dict:
    torch.manual_seed(SEED)
    model = Transformer(config, seq_len).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()

    init_vector = parameters_to_vector([p.detach().cpu() for p in model.parameters()])

    optimizer.zero_grad()
    logits = model(x.cuda())
    loss = loss_fn(logits, y.cuda())
    loss.backward()
    optimizer.step()

    final_vector = parameters_to_vector([p.detach().cpu() for p in model.parameters()])
    return {
        "init_vector": init_vector,
        "final_vector": final_vector,
        "loss": loss.item(),
    }


def run_piper_step(args: TestArgs, x: torch.Tensor, y: torch.Tensor) -> dict:
    torch.manual_seed(SEED)
    schedule = pp2_interleaved_1f1b_grid_schedule_mb1
    loss_fn = torch.nn.MSELoss()

    compiled = piper_setup(
        Transformer,
        (LLAMA_DEBUG, args.seq_len),
        torch.optim.Adam,
        [x],
        schedule,
        naive_gradient_sync=False,
    )

    actors = piper_metadata.actors
    init_parts = ray.get(
        [actor.get_stage_parameter_vectors.remote() for actor in actors.values()]
    )
    init_stage_to_vec = {}
    for stage_map in init_parts:
        init_stage_to_vec.update(stage_map)
    init_vector = _flatten_stage_param_vectors(init_stage_to_vec)

    ray.get(actors[0].load_input.remote([x]))
    ray.get(actors[len(actors) - 1].load_labels.remote(y))

    losses = piper_exec(compiled, schedule, [x], y, loss_fn, args.dp)

    final_parts = ray.get(
        [actor.get_stage_parameter_vectors.remote() for actor in actors.values()]
    )
    final_stage_to_vec = {}
    for stage_map in final_parts:
        final_stage_to_vec.update(stage_map)
    final_vector = _flatten_stage_param_vectors(final_stage_to_vec)

    return {
        "init_vector": init_vector,
        "final_vector": final_vector,
        "loss": losses[0][0],
        "raw_losses": losses,
    }


def main() -> None:
    assert (
        torch.cuda.is_available()
    ), "This test requires CUDA because Piper actors run on GPU."

    args = TestArgs()
    config = LLAMA_DEBUG
    batch_size = 2

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 1)
    x = torch.randint(
        0, config.vocab_size, (batch_size, args.seq_len), generator=generator
    )
    y = torch.randn((batch_size, args.seq_len, config.vocab_size), generator=generator)

    baseline = run_baseline_step(config, args.seq_len, x, y)

    ray.init(include_dashboard=False, log_to_driver=True, namespace="llama-parity")
    coordinator = PiperProgramCoordinator.remote(dp_degree=args.dp, pp_degree=args.pp)
    piper = ray.get(coordinator.run_program.remote(run_piper_step, args, x, y))[0]
    ray.shutdown()

    max_init_abs_diff = (
        (baseline["init_vector"] - piper["init_vector"]).abs().max().item()
    )
    loss_abs_diff = abs(baseline["loss"] - piper["loss"])
    max_updated_param_abs_diff = (
        (baseline["final_vector"] - piper["final_vector"]).abs().max().item()
    )

    print(f"baseline loss: {baseline['loss']}")
    print(f"piper loss: {piper['loss']}")
    print(f"piper raw losses: {piper['raw_losses']}")
    print(f"max initial abs diff: {max_init_abs_diff}")
    print(f"max loss abs diff: {loss_abs_diff}")
    print(f"max updated-parameter abs diff: {max_updated_param_abs_diff}")

    assert max_init_abs_diff < 1e-8
    assert loss_abs_diff < 1e-6
    assert max_updated_param_abs_diff < 1e-6


if __name__ == "__main__":
    main()
