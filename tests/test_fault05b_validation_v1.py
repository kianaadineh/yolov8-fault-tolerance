"""Local tests do not claim validation of the CUDA inference pipeline."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import hashlib

spec = importlib.util.spec_from_file_location(
    'fault05b_validation_v1',
    Path(__file__).resolve().parents[1] / 'src/fault05b_validation_v1.py',
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BitAndScopeTests(unittest.TestCase):
    def test_all_signed_values_and_bits_against_binary_encoding(self):
        for q in range(-128, 128):
            for bit in range(8):
                binary = list(format(q % 256, '08b'))
                position = 7 - bit
                binary[position] = '1' if binary[position] == '0' else '0'
                unsigned = int(''.join(binary), 2)
                expected = unsigned if unsigned < 128 else unsigned - 256
                self.assertEqual(m.flip_int8(q, bit), expected)

    def test_all_bit_flips_are_involutions_and_change_value(self):
        for q in range(-128, 128):
            for bit in range(8):
                changed = m.flip_int8(q, bit)
                self.assertNotEqual(changed, q)
                self.assertEqual(m.flip_int8(changed, bit), q)

    def test_sign_bit_boundaries(self):
        self.assertEqual(m.flip_int8(0, 7), -128)
        self.assertEqual(m.flip_int8(127, 7), -1)
        self.assertEqual(m.flip_int8(-127, 7), 1)

    def test_invalid_integer_and_bit_rejected(self):
        for value in (True, 1.0, -129, 128, '0'):
            with self.assertRaises(ValueError):
                m.flip_int8(value, 0)
        for bit in (True, 0.0, -1, 8, '7'):
            with self.assertRaises(ValueError):
                m.flip_int8(0, bit)

    def test_valid_subset(self):
        m.verify_subset(list(range(200)), list(range(1000)), list(range(1000, 5000)))

    def test_duplicate_wrong_size_and_overlap_rejected(self):
        for ids in ([0]*200, list(range(199)), list(range(199))+[1000]):
            with self.assertRaises(ValueError):
                m.verify_subset(ids, list(range(1000)), list(range(1000, 5000)))

    def test_source_checksum_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record'
            path.write_bytes(b'original')
            expected = hashlib.sha256(b'original').hexdigest()
            self.assertEqual(m.verified_bytes(path, expected), b'original')
            path.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                m.verified_bytes(path, expected)


if __name__ == '__main__':
    unittest.main()
