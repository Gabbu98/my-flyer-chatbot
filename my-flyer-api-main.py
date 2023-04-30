import os

from flask import Flask, request
from langchain import OpenAI
from llama_index import PromptHelper, GPTSimpleVectorIndex, LLMPredictor, download_loader

app = Flask(__name__)

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

@app.route('/chatbot',methods = ['GET'])
def chatbot():
   if request.method == 'POST':
      user = request.args['query']
      return "HEY"+user #redirect(url_for('success',name = user))
   else:
      query = request.args.get('query')
      response = index.query(query)
      return (response.__str__(), 200) #redirect(url_for('success',query = user))

if __name__ == '__main__':
   app.run(debug = True)