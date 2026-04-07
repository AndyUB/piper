"""
Tests for ZeRO-1/2/3 per-rank execution schedule correctness.

These tests verify the DAG transformation pipeline produces the correct
task ordering for the no-pp-4s schedule (4 stages, 1 rank, 1 microbatch)
with dp_degree=2 and 1 bucket per stage.

Assertions check the sorted-by-time_step execution sequence as a list of
(TaskType, stage_id, bucket_id) tuples.  Time_step *values* are intentionally
not tested — only relative ordering matters.
"""

import unittest
from src.piper_exec import TaskType, TaskDAG
from src.piper_graph_transform import (
    schedule_to_dag,
    expand_bucket_tasks,
    assign_time_steps,
    insert_p2p_ops,
    insert_ar_ops,
    insert_zero_ops,
)
from test.schedule_helpers import NO_PP_4STAGE_SCHEDULE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 1 bucket per each of the 4 stages on rank 0.
_BUCKET_COUNTS = {0: 1, 1: 1, 2: 1, 3: 1}


def _build_per_rank_dags(zero_stage: int) -> list[TaskDAG]:
    """Run the full DAG-construction pipeline for NO_PP_4STAGE_SCHEDULE.

    Mirrors the sequence in piper.py:
      schedule_to_dag → expand_bucket_tasks → assign_time_steps
      → insert_p2p_ops → insert_ar_ops → insert_zero_ops
    """
    dag = schedule_to_dag(NO_PP_4STAGE_SCHEDULE)
    dag = expand_bucket_tasks(dag, _BUCKET_COUNTS)
    assign_time_steps(dag)
    per_rank_dags = insert_p2p_ops(dag)
    # dp_degree=2: insert AR nodes for every (stage, bucket) pair.
    insert_ar_ops(per_rank_dags, trainable_bucket_keys=None)
    insert_zero_ops(per_rank_dags, zero_stage)
    return per_rank_dags


def _exec_seq(rank_dag: TaskDAG) -> list[tuple]:
    """Return the dispatch sequence for one rank as (TaskType, stage_id, bucket_id)."""
    sorted_nodes = sorted(rank_dag.nodes, key=lambda n: n.time_step)
    return [
        (n.task.type, n.task.batches[0].stage_id, n.bucket_id)
        for n in sorted_nodes
    ]


class TestZeroSchedule(unittest.TestCase):

    # -----------------------------------------------------------------------
    # ZeRO-1
    # -----------------------------------------------------------------------

    def test_zero1_exec_sequence(self):
        """ZeRO-1: post-UPD AG (s0 first) + per-stage AR; full params always resident.

        AR is dispatched at the same time_step as its trigger BWD so stable sort
        gives strict BWD→AR interleaving per stage.  AR is pre-queued on comm_stream
        immediately, allowing GPU-level overlap with the next BWD on comp_stream.

        Expected sequence
        -----------------
        Preamble (post-UPD AGs from previous iteration, s0 first):
          AG(s0) AG(s1) AG(s2) AG(s3)

        Main pipeline:
          FWD(s0) FWD(s1) FWD(s2) FWD(s3)
          BWD(s3) AR(s3)
          BWD(s2) AR(s2)
          BWD(s1) AR(s1)
          BWD(s0) AR(s0)
          UPD
        """
        per_rank_dags = _build_per_rank_dags(zero_stage=1)
        self.assertEqual(len(per_rank_dags), 1)
        seq = _exec_seq(per_rank_dags[0])

        AG  = TaskType.ALL_GATHER
        FWD = TaskType.FWD
        BWD = TaskType.BWD
        AR  = TaskType.ALL_REDUCE
        UPD = TaskType.UPD

        expected = [
            # Preamble: AGs for param reconstruction (s0 dispatched first)
            (AG,  0, -1),
            (AG,  1, -1),
            (AG,  2, -1),
            (AG,  3, -1),
            # Forward pass
            (FWD, 0, 0),
            (FWD, 1, 0),
            (FWD, 2, 0),
            (FWD, 3, 0),
            # Backward: strict BWD→AR interleaving
            (BWD, 3, 0), (AR, 3, 0),
            (BWD, 2, 0), (AR, 2, 0),
            (BWD, 1, 0), (AR, 1, 0),
            (BWD, 0, 0), (AR, 0, 0),
            # Optimizer step
            (UPD, 0, 0),
        ]
        self.assertEqual(seq, expected)

    # -----------------------------------------------------------------------
    # ZeRO-2
    # -----------------------------------------------------------------------

    def test_zero2_exec_sequence(self):
        """ZeRO-2: same post-UPD AGs as ZeRO-1, but AR → RS per BWD chunk.

        RS is at the same time_step as its trigger BWD (strict BWD→RS interleaving).
        ALLOC_FULL_GRADS fires one step before the first BWD of each chunk;
        FREE_FULL_GRADS fires at the same step as the last RS.

        For no-pp-4s every stage has exactly one BWD, so one RS per stage.

        Expected sequence
        -----------------
        Preamble:
          AG(s0) AG(s1) AG(s2) AG(s3)

        Main pipeline:
          FWD(s0) FWD(s1) FWD(s2) FWD(s3)
          ALLOC_GRADS(s3)
          BWD(s3) RS(s3) FREE_GRADS(s3)
          ALLOC_GRADS(s2)
          BWD(s2) RS(s2) FREE_GRADS(s2)
          ALLOC_GRADS(s1)
          BWD(s1) RS(s1) FREE_GRADS(s1)
          ALLOC_GRADS(s0)
          BWD(s0) RS(s0) FREE_GRADS(s0)
          UPD
        """
        per_rank_dags = _build_per_rank_dags(zero_stage=2)
        self.assertEqual(len(per_rank_dags), 1)
        seq = _exec_seq(per_rank_dags[0])

        AG   = TaskType.ALL_GATHER
        FWD  = TaskType.FWD
        BWD  = TaskType.BWD
        ALLG = TaskType.ALLOC_FULL_GRADS
        RS   = TaskType.REDUCE_SCATTER
        FREG = TaskType.FREE_FULL_GRADS
        UPD  = TaskType.UPD

        expected = [
            # Preamble
            (AG,   0, -1),
            (AG,   1, -1),
            (AG,   2, -1),
            (AG,   3, -1),
            # Forward pass
            (FWD,  0, 0),
            (FWD,  1, 0),
            (FWD,  2, 0),
            (FWD,  3, 0),
            # Backward: ALLOC→BWD→RS→FREE per stage
            (ALLG, 3, -1),
            (BWD,  3, 0), (RS, 3, 0), (FREG, 3, -1),
            (ALLG, 2, -1),
            (BWD,  2, 0), (RS, 2, 0), (FREG, 2, -1),
            (ALLG, 1, -1),
            (BWD,  1, 0), (RS, 1, 0), (FREG, 1, -1),
            (ALLG, 0, -1),
            (BWD,  0, 0), (RS, 0, 0), (FREG, 0, -1),
            (UPD,  0, 0),
        ]
        self.assertEqual(seq, expected)

    # -----------------------------------------------------------------------
    # ZeRO-3
    # -----------------------------------------------------------------------

    def test_zero3_exec_sequence(self):
        """ZeRO-3: strict per-stage sequential param lifecycle, no cross-stage prefetch.

        For no-pp-4s with 4 stages (s0-s3) and 1 microbatch per stage:
        - s3: FWD(t=3000) and BWD(t=4000) are adjacent (gap=1000=BUCKET_TIME_SCALE)
          → ONE param chunk covering both FWD and BWD.
        - s0,s1,s2: FWD and BWD are far apart → TWO param chunks each
          (one FWD-only chunk and one BWD-only chunk).

        Each param chunk:
          ALLOC_PARAMS(first.t-2) → AG_b0(first.t-1) → [compute] → FREE_PARAMS(last.t+1)

        Each BWD sub-chunk:
          ALLOC_GRADS(first_bwd.t-1) → BWD→RS→FREE_GRADS (all at same t)

        Expected sequence
        -----------------
          ALLP(s0) AG(s0,b0) FWD(s0) FREP(s0)
          ALLP(s1) AG(s1,b0) FWD(s1) FREP(s1)
          ALLP(s2) AG(s2,b0) FWD(s2) FREP(s2)
          ALLP(s3) AG(s3,b0) FWD(s3)
          ALLG(s3) BWD(s3) RS(s3) FREG(s3) FREP(s3)
          ALLP(s2) AG(s2,b0) ALLG(s2) BWD(s2) RS(s2) FREG(s2) FREP(s2)
          ALLP(s1) AG(s1,b0) ALLG(s1) BWD(s1) RS(s1) FREG(s1) FREP(s1)
          ALLP(s0) AG(s0,b0) ALLG(s0) BWD(s0) RS(s0) FREG(s0) FREP(s0)
          UPD
        """
        per_rank_dags = _build_per_rank_dags(zero_stage=3)
        self.assertEqual(len(per_rank_dags), 1)
        seq = _exec_seq(per_rank_dags[0])

        AG   = TaskType.ALL_GATHER
        FWD  = TaskType.FWD
        BWD  = TaskType.BWD
        ALLP = TaskType.ALLOC_FULL_PARAMS
        FREP = TaskType.FREE_FULL_PARAMS
        ALLG = TaskType.ALLOC_FULL_GRADS
        RS   = TaskType.REDUCE_SCATTER
        FREG = TaskType.FREE_FULL_GRADS
        UPD  = TaskType.UPD

        expected = [
            # s0 FWD-only param chunk
            (ALLP, 0, -1), (AG, 0, 0),
            (FWD,  0, 0),
            (FREP, 0, -1),
            # s1 FWD-only param chunk
            (ALLP, 1, -1), (AG, 1, 0),
            (FWD,  1, 0),
            (FREP, 1, -1),
            # s2 FWD-only param chunk
            (ALLP, 2, -1), (AG, 2, 0),
            (FWD,  2, 0),
            (FREP, 2, -1),
            # s3 combined FWD+BWD param chunk
            (ALLP, 3, -1), (AG, 3, 0),
            (FWD,  3, 0),
            (ALLG, 3, -1),
            (BWD,  3, 0), (RS, 3, 0), (FREG, 3, -1),
            (FREP, 3, -1),
            # s2 BWD-only param chunk
            (ALLP, 2, -1), (AG, 2, 0), (ALLG, 2, -1),
            (BWD,  2, 0), (RS, 2, 0), (FREG, 2, -1),
            (FREP, 2, -1),
            # s1 BWD-only param chunk
            (ALLP, 1, -1), (AG, 1, 0), (ALLG, 1, -1),
            (BWD,  1, 0), (RS, 1, 0), (FREG, 1, -1),
            (FREP, 1, -1),
            # s0 BWD-only param chunk
            (ALLP, 0, -1), (AG, 0, 0), (ALLG, 0, -1),
            (BWD,  0, 0), (RS, 0, 0), (FREG, 0, -1),
            (FREP, 0, -1),
            (UPD,  0, 0),
        ]
        self.assertEqual(seq, expected)


if __name__ == "__main__":
    unittest.main()
