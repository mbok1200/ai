import unittest
from src.models.ollama_client import OllamaClient

class TestOllamaClient(unittest.TestCase):

    def setUp(self):
        self.client = OllamaClient()
        self.client.initialize()

    def test_load_model(self):
        model_name = "test_model"
        result = self.client.load_model(model_name)
        self.assertTrue(result)

    def test_generate_response(self):
        input_data = "Hello, how are you?"
        response = self.client.generate_response(input_data)
        self.assertIsInstance(response, str)

if __name__ == '__main__':
    unittest.main()