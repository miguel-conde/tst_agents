import unittest
from circular_memory import CircularMemory, CircularMemoryWithBuffer

class TestCircularMemory(unittest.TestCase):

    def test_initialization(self):
        buffer = CircularMemory(5)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.size, 5)

        with self.assertRaises(ValueError):
            CircularMemory(0)

    def test_add_and_get(self):
        buffer = CircularMemory(3)
        buffer.add(1)
        buffer.add(2)
        buffer.add(3)
        self.assertEqual(buffer.get(), [1, 2, 3])

        # Overwrite oldest element
        buffer.add(4)
        self.assertEqual(buffer.get(), [2, 3, 4])

        buffer.add(5)
        self.assertEqual(buffer.get(), [3, 4, 5])

    def test_is_full(self):
        buffer = CircularMemory(2)
        self.assertFalse(buffer.is_full())
        buffer.add(1)
        self.assertFalse(buffer.is_full())
        buffer.add(2)
        self.assertTrue(buffer.is_full())

    def test_clear(self):
        buffer = CircularMemory(3)
        buffer.add(1)
        buffer.add(2)
        buffer.clear()
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.get(), [])

    def test_iteration(self):
        buffer = CircularMemory(3)
        buffer.add(1)
        buffer.add(2)
        buffer.add(3)
        buffer.add(4)  # Overwrites oldest (1)
        self.assertEqual(list(buffer), [2, 3, 4])

    def test_getitem_and_setitem(self):
        buffer = CircularMemory(3)
        buffer.add(1)
        buffer.add(2)
        buffer.add(3)
        self.assertEqual(buffer[0], 1)
        self.assertEqual(buffer[1], 2)
        self.assertEqual(buffer[2], 3)

        buffer[1] = 42
        self.assertEqual(buffer[1], 42)

        with self.assertRaises(IndexError):
            _ = buffer[3]

        with self.assertRaises(IndexError):
            buffer[3] = 99

    def test_overwritten_elements(self):
        buffer = CircularMemory(3)
        self.assertIsNone(buffer.add(1))
        self.assertIsNone(buffer.add(2))
        self.assertIsNone(buffer.add(3))
        self.assertEqual(buffer.add(4), 1)
        self.assertEqual(buffer.add(5), 2)
        self.assertEqual(buffer.add(6), 3)

class TestCircularMemoryWithBuffer(unittest.TestCase):

    def summarize_fn(self, buffer, *args):
        return "SUMMARIZED: " + ", ".join(map(str, buffer))

    def test_initialization(self):
        memory_with_buffer = CircularMemoryWithBuffer(5, 3, self.summarize_fn)
        self.assertEqual(memory_with_buffer.size, 5)
        self.assertEqual(memory_with_buffer.buffer_size, 3)
        self.assertEqual(len(memory_with_buffer.memory), 0)
        self.assertEqual(len(memory_with_buffer.buffer), 0)

        with self.assertRaises(ValueError):
            CircularMemoryWithBuffer(0, 3, self.summarize_fn)
        with self.assertRaises(ValueError):
            CircularMemoryWithBuffer(5, 0, self.summarize_fn)

    def test_add_to_memory(self):
        memory_with_buffer = CircularMemoryWithBuffer(3, 3, self.summarize_fn)
        memory_with_buffer.add(1)
        memory_with_buffer.add(2)
        memory_with_buffer.add(3)
        self.assertEqual(memory_with_buffer.get_memory(), [1, 2, 3])

        memory_with_buffer.add(4)  # Overwrites 1
        self.assertEqual(memory_with_buffer.get_memory(), [2, 3, 4])
        self.assertEqual(memory_with_buffer.get_buffer(), [1])

    def test_add_to_buffer_and_summarize(self):
        memory_with_buffer = CircularMemoryWithBuffer(3, 2, self.summarize_fn)
        memory_with_buffer.add(1)
        memory_with_buffer.add(2)
        memory_with_buffer.add(3)
        memory_with_buffer.add(4)  # Overwrites 1
        memory_with_buffer.add(5)  # Overwrites 2, buffer full

        # Buffer full, summarize should trigger
        self.assertEqual(memory_with_buffer.get_memory(), [3, 4, 5])
        self.assertEqual(memory_with_buffer.get_buffer(), [])

    def test_get_all(self):
        memory_with_buffer = CircularMemoryWithBuffer(3, 3, self.summarize_fn)
        memory_with_buffer.add(1)
        memory_with_buffer.add(2)
        memory_with_buffer.add(3)
        memory_with_buffer.add(4)  # Overwrites 1
        memory_with_buffer.add(5)  # Overwrites 2
        self.assertEqual(memory_with_buffer.get_all(), [1, 2, 3, 4, 5])

if __name__ == "__main__":
    unittest.main()