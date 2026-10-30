import torch


def flip_int8_bit(q_tensor, index, bit_position):
    """Flip one bit in a copy of a signed INT8 tensor."""
    if q_tensor.dtype != torch.int8:
        raise TypeError("Expected torch.int8 tensor")

    if not 0 <= bit_position <= 7:
        raise ValueError("bit_position must be between 0 and 7")

    faulty = q_tensor.clone()

    before = int(q_tensor[index].item())
    before_bits = before & 0xFF

    after_bits = before_bits ^ (1 << bit_position)
    after = after_bits if after_bits < 128 else after_bits - 256

    faulty[index] = after

    record = {
        "index": tuple(index),
        "bit_position": bit_position,
        "value_before": before,
        "value_after": after,
        "bits_before": format(before_bits, "08b"),
        "bits_after": format(after_bits, "08b"),
    }

    return faulty, record
