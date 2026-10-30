
import torch


def flip_int4_bit(value, bit_position):
    """Flip one bit in a signed 4-bit two's-complement value."""
    if not isinstance(value, int) or not -8 <= value <= 7:
        raise ValueError("INT4 value must be an integer in [-8, 7]")
    if not isinstance(bit_position, int) or not 0 <= bit_position <= 3:
        raise ValueError("bit_position must be an integer in [0, 3]")

    before_bits = value & 0xF
    after_bits = before_bits ^ (1 << bit_position)
    after = after_bits if after_bits < 8 else after_bits - 16

    record = {
        "value_before": value,
        "value_after": after,
        "bit_position": bit_position,
        "bits_before": format(before_bits, "04b"),
        "bits_after": format(after_bits, "04b"),
    }
    return after, record


def flip_int4_tensor_bit(q_tensor, index, bit_position):
    """Flip one bit in a copy of a logical signed INT4 tensor."""
    if q_tensor.dtype != torch.int8:
        raise TypeError("Logical INT4 tensor must use torch.int8 storage")
    if not torch.all((q_tensor >= -8) & (q_tensor <= 7)).item():
        raise ValueError("Tensor contains values outside signed INT4 range")

    faulty = q_tensor.clone()
    before = int(q_tensor[index].item())
    after, record = flip_int4_bit(before, bit_position)
    faulty[index] = after
    record["index"] = tuple(index)

    return faulty, record
