from google import genai


class GeminiClient:
    """
    Client for interacting with Google's Gemini API.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key (str):
                Gemini API key.

            model_name (str):
                Gemini model name.
        """

        if not api_key:
            raise ValueError("Gemini API key is missing.")

        self.model_name = model_name

        try:
            self.client = genai.Client(
                api_key=api_key,
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Gemini client: {e}"
            ) from e

    def generate_response(
        self,
        prompt: str,
    ):
        """
        Generate a response from Gemini.

        Args:
            prompt (str):
                Prompt to send to Gemini.

        Returns:
            GenerateContentResponse:
                Raw Gemini response object.
        """

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        try:

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            return response

        except Exception as e:
            raise RuntimeError(
                f"Failed to generate response: {e}"
            ) from e

    def test_connection(self) -> bool:
        """
        Test whether Gemini is reachable.

        Returns:
            bool:
                True if the connection succeeds.
        """

        try:

            self.client.models.generate_content(
                model=self.model_name,
                contents="Hello",
            )

            return True

        except Exception:
            return False