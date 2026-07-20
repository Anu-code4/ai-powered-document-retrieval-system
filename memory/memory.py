
"""
Conversation Memory

Stores the recent conversation between the
user and the AI assistant.
"""


class ConversationMemory:

    def __init__(self, max_history=5):

        self.max_history = max_history

        self.history = []


    def add_user_message(self, message):

        self.history.append(
            {
                "role": "user",
                "content": message
            }
        )

        self._trim_history()


    def add_ai_message(self, message):

        self.history.append(
            {
                "role": "assistant",
                "content": message
            }
        )

        self._trim_history()


    def get_history(self):

        return self.history


    def clear(self):

        self.history = []


    def _trim_history(self):

        while len(self.history) > self.max_history * 2:

            self.history.pop(0)


if __name__ == "__main__":
    memory = ConversationMemory()

    memory.add_user_message("Hello")
    memory.add_ai_message("Hi")
    memory.add_user_message("What is Chunking?")

    print(memory.get_history())