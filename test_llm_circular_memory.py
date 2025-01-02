import unittest
from unittest.mock import MagicMock
from llm_circular_memory import ChatCircularMemory, ConversationMessage

class TestChatCircularMemory(unittest.TestCase):

    def setUp(self):
        self.chat_memory = ChatCircularMemory(
            sys_prompt="Test system prompt",
            short_term_memory_size=5,
            short_term_buffer_size=2,
            mid_term_memory_size=3,
            mid_term_buffer_size=2
        )

        # Mock the OpenAI client
        self.chat_memory.client = MagicMock()
        self.chat_memory.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Mocked summary"))]
        )

    def test_initialization(self):
        self.assertEqual(self.chat_memory.sys_prompt, "Test system prompt")
        self.assertEqual(len(self.chat_memory.long_term_memory), 1)
        self.assertEqual(self.chat_memory.short_term_memory.size, 5)
        self.assertEqual(self.chat_memory.short_term_memory.buffer_size, 2)
        self.assertEqual(self.chat_memory.mid_term_memory.size, 3)
        self.assertEqual(self.chat_memory.mid_term_memory.buffer_size, 2)

    def test_add_message_to_short_term_memory(self):
        message = ConversationMessage(role="user", content="Hello, world!")
        self.chat_memory.add_message(message)
        self.assertIn(message, self.chat_memory.short_term_memory.get_memory())

    def test_summarize(self):
        buffer = [
            ConversationMessage(role="user", content="Message 1"),
            ConversationMessage(role="user", content="Message 2")
        ]
        summary = self.chat_memory.summarize(buffer)
        self.assertEqual(summary.content, "Mocked summary")

    def test_summary_buffer(self):
        buffer = [
            ConversationMessage(role="user", content="Message 1"),
            ConversationMessage(role="user", content="Message 2")
        ]
        self.chat_memory.summary_buffer(buffer, self.chat_memory.mid_term_memory)
        self.assertIn(
            ConversationMessage(role="system", content="Mocked summary"),
            self.chat_memory.mid_term_memory.get_memory()
        )

    def test_summary_lt_memory(self):
        buffer = [
            ConversationMessage(role="user", content="Message 1"),
            ConversationMessage(role="user", content="Message 2")
        ]
        updated_lt_memory = self.chat_memory.summary_lt_memory(buffer, self.chat_memory.long_term_memory)
        self.assertEqual(len(updated_lt_memory), 2)
        self.assertEqual(updated_lt_memory[1].content, "Mocked summary")

    def test_get_all(self):
        user_message = ConversationMessage(role="user", content="Hello")
        self.chat_memory.add_message(user_message)
        all_messages = self.chat_memory.get_all()
        self.assertIn(user_message, all_messages)
        self.assertIn(self.chat_memory.long_term_memory[0], all_messages)

if __name__ == "__main__":
    unittest.main()
