import unittest
from unittest.mock import MagicMock
from llm_handler import ChatGPTRequester, ConversationMessage

class TestChatGPTRequester(unittest.TestCase):

    def setUp(self):
        self.mock_client = MagicMock()
        self.requester = ChatGPTRequester(
            client=self.mock_client,
            model="gpt-4o",
            embeddings_model="text-embedding-ada-002",
            retries=3,
            service_unavailable_wait_secs=0.1,
            temperature=0.7
        )

    def test_reset_token_counters(self):
        self.requester.prompt_tokens = 50
        self.requester.completion_tokens = 30
        self.requester.total_tokens = 80

        self.requester.reset_token_counters()

        self.assertEqual(self.requester.prompt_tokens, 0)
        self.assertEqual(self.requester.completion_tokens, 0)
        self.assertEqual(self.requester.total_tokens, 0)

    def test_request_successful(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response content"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        self.mock_client.chat.completions.create.return_value = mock_response

        messages = [ConversationMessage(role="user", content="Hello").model_dump()]
        result = self.requester.request(messages)

        self.assertEqual(result.content, "Response content")
        self.assertEqual(self.requester.prompt_tokens, 10)
        self.assertEqual(self.requester.completion_tokens, 20)
        self.assertEqual(self.requester.total_tokens, 30)

    # def test_request_with_retries(self):
    #     self.mock_client.chat.completions.create.side_effect = [
    #         Exception("Temporary failure"),
    #         MagicMock(
    #             choices=[MagicMock(message=MagicMock(content="Recovered content"))],
    #             usage=MagicMock(prompt_tokens=5, completion_tokens=15, total_tokens=20)
    #         )
    #     ]

    #     messages = [ConversationMessage(role="user", content="Hello").model_dump()]
    #     result = self.requester.request(messages)

    #     self.assertEqual(result.content, "Recovered content")
    #     self.assertEqual(self.requester.prompt_tokens, 5)
    #     self.assertEqual(self.requester.completion_tokens, 15)
    #     self.assertEqual(self.requester.total_tokens, 20)

    # def test_request_embedding_successful(self):
    #     mock_response = MagicMock()
    #     mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    #     mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=0, total_tokens=5)
    #     self.mock_client.embeddings.create.return_value = mock_response

    #     result = self.requester.request_embedding("Test text")

    #     self.assertEqual(result, [0.1, 0.2, 0.3])
    #     self.assertEqual(self.requester.prompt_tokens, 10)
    #     self.assertEqual(self.requester.total_tokens, 10)

    # def test_request_embedding_with_retries(self):
    #     self.mock_client.embeddings.create.side_effect = [
    #         Exception("Temporary failure"),
    #         MagicMock(
    #             data=[MagicMock(embedding=[0.4, 0.5, 0.6])],
    #             usage=MagicMock(prompt_tokens=7, completion_tokens=0, total_tokens=7)
    #         )
    #     ]

    #     result = self.requester.request_embedding("Another test text")

    #     self.assertEqual(result, [0.4, 0.5, 0.6])
    #     self.assertEqual(self.requester.prompt_tokens, 7)
    #     self.assertEqual(self.requester.total_tokens, 7)

if __name__ == "__main__":
    unittest.main()
