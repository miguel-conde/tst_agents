from typing import List, Optional, Iterator, Callable, Any

class CircularMemory:
    """
    CircularMemory implements a circular buffer that stores a fixed number of elements.
    Once the buffer is full, new elements overwrite the oldest ones in the buffer.
    
    Attributes:
        size (int): The maximum number of elements the buffer can hold.
        memory (List): The list used to store elements in the buffer.
        idx (int): The current index in the circular buffer where the next element will be added. It also indicates the position of the oldest element in the buffer.
    """
    def __init__(self, size: int):
        """
        Initializes the CircularMemory with a fixed size.

        Args:
            size (int): The maximum number of elements the buffer can hold.
        """
        if size <= 0:
            raise ValueError("Size of the buffer must be a positive integer.")
        self.size = size
        self.memory: List = []
        self.idx: int = 0

    def add(self, item: object) -> Optional[object]:
        """
        Adds an item to the circular buffer. If the buffer is full, overwrites the oldest element.

        Args:
            item (object): The item to be added to the buffer.

        Returns:
            Optional[object]: The overwritten element if the buffer is full, otherwise None.
        """
        if len(self.memory) < self.size:
            self.memory.append(item)
            return None
        else:
            out = self.memory[self.idx]
            self.memory[self.idx] = item
            self.idx = self._next_index(self.idx)
            return out

    def get(self) -> List[object]:
        """
        Retrieves the current state of the buffer, ordered by age (oldest first).

        Returns:
            List[object]: The list of elements in the buffer, starting from the oldest.
        """
        if len(self.memory) < self.size:
            return self.memory
        return self.memory[self.idx:] + self.memory[:self.idx]

    def is_full(self) -> bool:
        """
        Checks if the buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return len(self.memory) == self.size

    def clear(self) -> None:
        """
        Clears all elements from the buffer.
        """
        self.memory = []
        self.idx = 0

    def _next_index(self, index: int) -> int:
        """
        Calculates the next index in a circular manner.

        Args:
            index (int): The current index.

        Returns:
            int: The next index.
        """
        return (index + 1) % self.size

    def __iter__(self) -> Iterator[object]:
        """
        Returns an iterator over the buffer in order of age (oldest first).

        Returns:
            Iterator[object]: An iterator over the buffer elements.
        """
        return iter(self.get())

    def __str__(self) -> str:
        """
        Returns a string representation of the buffer, ordered by age (oldest first).

        Returns:
            str: String representation of the buffer, starting from the oldest element.
        """
        return str(self.get())

    def __repr__(self) -> str:
        """
        Returns the official string representation of the buffer, ordered by age (oldest first).

        Returns:
            str: Official string representation of the buffer, starting from the oldest element.
        """
        return str(self.get())

    def __len__(self) -> int:
        """
        Returns the number of elements currently in the buffer.

        Returns:
            int: Number of elements in the buffer.
        """
        return len(self.memory)

    def __getitem__(self, idx: int) -> object:
        """
        Retrieves the element at the specified index in the buffer.

        Args:
            idx (int): Index of the element to retrieve.

        Returns:
            object: The element at the specified index.
        """
        if idx < 0 or idx >= len(self.memory):
            raise IndexError("Index out of range.")
        return self.memory[idx]

    def __setitem__(self, idx: int, value: object) -> None:
        """
        Sets the element at the specified index in the buffer.

        Args:
            idx (int): Index of the element to set.
            value (object): The value to set at the specified index.
        """
        if idx < 0 or idx >= len(self.memory):
            raise IndexError("Index out of range.")
        self.memory[idx] = value

class CircularMemoryWithBuffer:
    """
    CircularMemoryWithBuffer extends the functionality of CircularMemory by adding a buffer for handling
    overwritten elements. When the buffer reaches its capacity, a summarization function is applied.

    Attributes:
        size (int): The maximum number of elements in the primary memory.
        buffer_size (int): The maximum number of elements in the buffer.
        memory (CircularMemory): The primary circular memory.
        buffer (CircularMemory): The buffer for storing overwritten elements.
        summarize_fn (Callable): The function to summarize the buffer when it is full.
        summarize_fn_args (List): Additional arguments for the summarize function.
    """

    def __init__(self, size: int, buffer_size: int, summarize_fn: Callable, summarize_fn_args: List[Any] = []):
        """
        Initializes the CircularMemoryWithBuffer with specified sizes and a summarization function.

        Args:
            size (int): The size of the primary circular memory.
            buffer_size (int): The size of the buffer.
            summarize_fn (Callable): The function to summarize the buffer when it is full.
            summarize_fn_args (List): Additional arguments for the summarize function.
        """
        if size <= 0 or buffer_size <= 0:
            raise ValueError("Size and buffer_size must be positive integers.")
        self.size = size
        self.memory = CircularMemory(size)
        self.buffer_size = buffer_size
        self.buffer = CircularMemory(buffer_size)
        self.summarize_fn = summarize_fn
        self.summarize_fn_args = summarize_fn_args

    def add(self, item: Any) -> Any:
        """
        Adds an item to the primary memory. If an element is overwritten, it is added to the buffer.
        If the buffer becomes full, the summarization function is applied.

        Args:
            item (Any): The item to be added to the primary memory.

        Returns:
            Any: The element that was overwritten in the primary memory, if any.
        """
        overwritten_element = self.memory.add(item)
        if overwritten_element is not None:
            self.buffer.add(overwritten_element)
        if len(self.buffer) == self.buffer_size:
            summarized = self.summarize_fn(self.buffer.get(), *self.summarize_fn_args)
            print(f"Buffer is full: {self.buffer.get()} - SUMMARIZING -> {summarized}")
            self.buffer.clear()
        return overwritten_element

    def get_memory(self) -> List[Any]:
        """
        Retrieves the current state of the primary memory.

        Returns:
            List[Any]: The elements in the primary memory, ordered by age.
        """
        return self.memory.get()

    def get_buffer(self) -> List[Any]:
        """
        Retrieves the current state of the buffer.

        Returns:
            List[Any]: The elements in the buffer, ordered by age.
        """
        return self.buffer.get()

    def get_all(self) -> List[Any]:
        """
        Retrieves all elements from the primary memory and buffer, ordered by age.

        Returns:
            List[Any]: All elements from memory and buffer, ordered by age.
        """
        return self.buffer.get() + self.memory.get()

    def __str__(self) -> str:
        """
        Returns a string representation of the primary memory and the buffer.

        Returns:
            str: String representation of the memory and buffer.
        """
        return f"Memory: {self.memory} | Buffer: {self.buffer}"

    def __repr__(self) -> str:
        """
        Returns the official string representation of the primary memory and the buffer.

        Returns:
            str: Official string representation of the memory and buffer.
        """
        return f"Memory: {repr(self.memory)} | Buffer: {repr(self.buffer)}"

    def __len__(self) -> int:
        """
        Returns the total number of elements in the primary memory and the buffer.

        Returns:
            int: Total number of elements in the memory and buffer.
        """
        return len(self.memory) + len(self.buffer)

# Example usage
if __name__ == '__main__':
    from circular_memory_old import CircularMemoryWithBuffer

    def summmary_fn_lt(buffer, last_memory):
        summarized = "-".join([str(x) for x in buffer])
        last_memory[0] = last_memory[0] + "-" + summarized
        return last_memory

    def summmary_fn_mt(buffer, long_term_memory):
        summarized =  "-".join([str(x) for x in buffer])
        long_term_memory.add(summarized)
        return summarized

    def summmary_fn_st(buffer, mid_term_memory):
        summarized =  "-".join([str(x) for x in buffer])
        mid_term_memory.add(summarized)
        return summarized

    last_memory = [""]
    long_term_memory = CircularMemoryWithBuffer(5, 5, summarize_fn=summmary_fn_lt, summarize_fn_args=[last_memory])
    mid_term_memory = CircularMemoryWithBuffer(5, 5, summarize_fn=summmary_fn_mt, summarize_fn_args=[long_term_memory])
    short_term_memory = CircularMemoryWithBuffer(5, 5, summarize_fn=summmary_fn_st, summarize_fn_args=[mid_term_memory])

    for i in range(1000):
        short_term_memory.add(i)
        print(short_term_memory)