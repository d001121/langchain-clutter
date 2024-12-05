import openai, os

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from Qwen import Qwen
from langchain.memory import ConversationBufferMemory


model_path = "/workspace/langchain-clutter/models/Qwen2.5-3B-Instruct-GPTQ-Int8"

llm = Qwen()
llm.load_model(model_path)
# llm = OpenAIChat(max_tokens=2048, temperature=0.5)
question1 = "请把下面这句话翻译成英文： \n\n {question}?"
q1_prompt = PromptTemplate(template=question1, input_variables=["question"])
q1_chain = LLMChain(llm=llm, prompt=q1_prompt, output_key="english_question")

question2 = "{english_question}"
q2_prompt = PromptTemplate(template=question2, input_variables=["english_question"])
q2_chain = LLMChain(llm=llm, prompt=q2_prompt, output_key="english_answer")

question3 = "请把下面这一段翻译成中文： \n\n{english_answer}?"
q3_prompt = PromptTemplate(template=question3, input_variables=["english_answer"])
q3_chain = LLMChain(llm=llm, prompt=q3_prompt)


simple_qa_chain=SimpleSequentialChain(
    chains=[q1_chain,q2_chain,q3_chain], input_key="question" ,verbose=True)

answer=simple_qa_chain.run(question="请你作为一个机器学习的专家，介绍一下CNN的原理。")
print(answer)



# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("/root/chatglm3-6b-model", trust_remote_code=True)
# model = AutoModel.from_pretrained("/root/chatglm3-6b-model", load_in_8bit=True, device_map='cuda', trust_remote_code=True, llm_int8_enable_fp32_cpu_offload=True).eval()

# response, history = model.chat(tokenizer, "你好，我叫李华", history=[])
# print(response)

# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)

# response, history = model.chat(tokenizer, "我叫什么名字？", history=history)
# print(response)