from langchain_community.chat_models import ChatAnyscale
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ChatMessageHistory
from queue import Queue
from threading import Thread
import sys

INPUTMARKER_END = "-- END --"
ANYSCALE_ENDPOINT_TOKEN = ""

class LangchainChatAgent():

    class StreamingCBH(BaseCallbackHandler):
        def __init__(self, q):
            self.q = q

        def on_llm_new_token(
            self,
            token,
            *,
            run_id,
            parent_run_id = None,
            **kwargs,
        ) -> None:
            self.q.put(token)

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            self.q.put(INPUTMARKER_END)


    def __init__(self, model: str = None):
        # This simple example doesn't modify the past conversation.
        # Eventually you run out of context window, but this should be enough
        # for a 30-step conversation.
        # You need to either trim the message history or summarize it for longer conversations.
        self.message_history = ChatMessageHistory()
        self.model = model
        self.llm = ChatAnyscale(anyscale_api_key=ANYSCALE_ENDPOINT_TOKEN,
                  temperature=0, model_name=self.model, streaming=True)


    def process_input(self, user_message: str):
        self.message_history.add_user_message(user_message)
        myq = Queue()
        
        thread =  Thread(target = self.llm.invoke, kwargs =
                        {'input': self.message_history.messages,
                         'config': {'callbacks':[self.StreamingCBH(myq)]}}
                   )
        thread.start()
        ai_message = ''
        while True:
            token = myq.get()
            if token == INPUTMARKER_END:
                break
            ai_message += token
            yield token

        self.message_history.add_ai_message(ai_message)

agent = LangchainChatAgent("meta-llama/Meta-Llama-3-8B-Instruct")
sys.stdout.write("Let's have a chat. (Enter `quit` to exit)\n")
while True:
    sys.stdout.write('> ')
    inp = input()
    if inp == 'quit':
        break
    for word in agent.process_input(inp):
        sys.stdout.write(word)
        sys.stdout.flush()
    sys.stdout.write('\n')