import numpy as np


def get_positional_encoding(seq_len, d_model):
    assert d_model % 2 == 0, "d_model must be even"

    pos = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]    # (1, d_model)

    # Compute the angle rates: 1 / (10000^(2i/d_model))
    angle_rates = 1 / np.power(10000, (i // 2 * 2) / d_model)

    angle_rads = pos * angle_rates # (seq_len, d_model

    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads

d_model = 512
seq_len = 11

pe = get_positional_encoding(seq_len, d_model)

# i. PE at position 10, dimension index 6
value1 = pe[10, 6]

# ii. PE at position 8, dimension index 7
value2 = pe[8, 7]

print(f"Positional Encoding at position 10, dimension 6: {value1}")
print(f"Positional Encoding at position 8, dimension 7: {value2}")