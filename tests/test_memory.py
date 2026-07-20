

from memory.memory import ConversationMemory


def test_memory_initialization():
    memory = ConversationMemory()

    assert memory.max_history == 5
    assert memory.get_history() == []


def test_add_user_message():
    memory = ConversationMemory()

    memory.add_user_message("Hello")

    history = memory.get_history()

    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"


def test_add_ai_message():
    memory = ConversationMemory()

    memory.add_ai_message("Hi")

    history = memory.get_history()

    assert len(history) == 1
    assert history[0]["role"] == "assistant"
    assert history[0]["content"] == "Hi"


def test_clear_memory():
    memory = ConversationMemory()

    memory.add_user_message("Hello")
    memory.add_ai_message("Hi")

    memory.clear()

    assert memory.get_history() == []


def test_history_limit():
    memory = ConversationMemory(max_history=2)

    memory.add_user_message("Q1")
    memory.add_ai_message("A1")

    memory.add_user_message("Q2")
    memory.add_ai_message("A2")

    memory.add_user_message("Q3")
    memory.add_ai_message("A3")

    history = memory.get_history()

    # Only last 2 conversations should remain
    assert len(history) == 4

    assert history[0]["content"] == "Q2"
    assert history[1]["content"] == "A2"
    assert history[2]["content"] == "Q3"
    assert history[3]["content"] == "A3"
    