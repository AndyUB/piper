import tempfile
import unittest
from pathlib import Path

import torch

from scripts.compare_p2p_tensors import compare_logged_tensors


class TestP2PTensorCompare(unittest.TestCase):
    def _write_tensor_log(self, folder: Path, direction: str, key: dict, tensor: torch.Tensor):
        payload = {
            "meta": {**key, "direction": direction, "global_rank": 0},
            "tensor": tensor,
        }
        file_name = f"{direction}__{key['pass']}__s{key['src_stage']}__d{key['dst_stage']}__mb{key['mb_idx']}__t{key['tensor_idx']}.pt"
        torch.save(payload, folder / file_name)

    def test_compare_logged_tensors_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            key = {
                "pass": "fwd",
                "src_stage": 0,
                "dst_stage": 1,
                "mb_idx": 0,
                "tensor_idx": 0,
                "dp_rank": 0,
            }
            tensor = torch.tensor([1.0, 2.0, 3.0])
            self._write_tensor_log(folder, "send", key, tensor)
            self._write_tensor_log(folder, "recv", key, tensor.clone())

            checked_pairs, errors = compare_logged_tensors(str(folder))

            self.assertEqual(checked_pairs, 1)
            self.assertEqual(errors, [])

    def test_compare_logged_tensors_reports_max_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            key = {
                "pass": "bwd",
                "src_stage": 1,
                "dst_stage": 0,
                "mb_idx": 2,
                "tensor_idx": 0,
                "dp_rank": 0,
            }
            self._write_tensor_log(folder, "send", key, torch.tensor([1.0, 2.0]))
            self._write_tensor_log(folder, "recv", key, torch.tensor([1.0, 2.5]))

            checked_pairs, errors = compare_logged_tensors(str(folder))

            self.assertEqual(checked_pairs, 1)
            self.assertEqual(len(errors), 1)
            self.assertIn("max_diff=5.00000000e-01", errors[0])


if __name__ == "__main__":
    unittest.main()
