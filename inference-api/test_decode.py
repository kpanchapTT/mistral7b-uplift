from pathlib import Path

import torch
from decode_backend_v1 import batch_top_pk_logits_efficient
from tt_models.falcon40b.decode_v0 import top_pk_logits_efficient


def test_decode():
    file_dir = Path(__file__).resolve().parent
    data_dir = Path(file_dir / "test_data")
    before_top_pk_output = torch.load(data_dir / "before_top_pk_output.pt")
    # test equal when all same params
    top_p = [0.9] * 32
    top_k = [10] * 32
    temperature = [1.1] * 32
    r1 = batch_top_pk_logits_efficient(before_top_pk_output, top_p, top_k, temperature)
    r2 = top_pk_logits_efficient(
        before_top_pk_output, top_p[0], top_k[0], temperature[0]
    )
    assert torch.equal(r1, r2)


if __name__ == "__main__":
    test_decode()
