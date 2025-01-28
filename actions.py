from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Initialize models
retriever = SentenceTransformer("intfloat/multilingual-e5-base")
generator_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
generator_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B-Instruct", device_map="auto")

def generate_response(prompt):
    inputs = generator_tokenizer(prompt, return_tensors="pt").to(generator_model.device)
    outputs = generator_model.generate(**inputs, max_length=512)
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

def system_analysis(question):
    prompt = f"Decompose into sub-queries:\n{question}\nOutput:"
    return generate_response(prompt)

def direct_answer(question):
    return generate_response(question)

def retrieval_answer(question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    return generate_response(prompt)

def query_transformation(question, history):
    prompt = f"Rephrase query (History: {history}):\n{question}\nOutput:"
    return generate_response(prompt)

def summary_answer(question, contexts):
    prompt = f"Summarize answer from contexts:\n{contexts}\nQuestion: {question}\nFinal Answer:"
    return generate_response(prompt)