# This is a sample Python script.
import os

from langchain import OpenAI
from llama_index import download_loader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def chatbot(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    os.environ['OPENAI_API_KEY'] = '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'

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

    # load docs and index them
    loader = SimpleDirectoryReader('./data')
    documents = loader.load_data()
    index = GPTSimpleVectorIndex(documents)

    # Save your index to index.json
    index.save_to_disk('index.json')
    # # Load the index from your saved index.json file
    # index = GPTSimpleVectorIndex.load_from_disk('index.json')

    # define anoter LLM explicitly
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    # define prompt configuraiton
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    last_token_usage = index.llm_predictor.last_token_usage

    print(f"last_token_usage={last_token_usage}")

    # Querying the index
    while True:
        prompt = input("Type prompt...")
        response = index.query(prompt)
        print(response)


# Press the green button in the gutter to run the script.
if __name__ == '__mai2n__':
    chatbot('MyFlyer Chatbot')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
