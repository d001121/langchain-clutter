# import openai, os
# from Qwen import Qwen
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# model_path = "../models/Qwen2.5-Coder-0.5B"
# llm = Qwen()
# llm.load_model(model_path)
# multiple_choice="""
# 请针对 >>> 和 <<< 中间的用户问题，选择一个合适的工具去回答它的问题。只要用A、B、C的选项字母告诉我答案。
# 如果你觉得都不合适，就选D。
# >>>{question}<<<
# 我们有的工具包括：
# A. 一个能够查询商品信息，为用户进行商品导购的工具
# B. 一个能够查询订单信息，获得最新的订单情况的工具
# C. 一个能够搜索商家的退换货政策、运费、物流时长、支付渠道、覆盖国家的工具
# D. 都不合适
# """
# multiple_choice_prompt=PromptTemplate(template=multiple_choice,input_variables=["question"])
# choice_chain=LLMChain(llm=llm,prompt=multiple_choice_prompt, output_key="answer")
# answer=choice_chain.run(question="我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？")
# print(answer)



from tempfile import template
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
# from ChatGLM3 import ChatGLM3
from langchain.chains import LLMChain
from Internlm import Internlm

 
def write_unit_test(function_to_test,model_path,unit_test_package="pytest"):  
    # 解释源代码的步骤
    explain_code = """"# How to write great unit tests with {unit_test_package}
    In this advanced tutorial for experts, we'll use Python 3.10 and `{unit_test_package}` to write a suite of unit tests to verify the behavior of the following function.
    ```python
    {function_to_test}
    ```
    Before writing any unit tests, let's review what each element of the function is doing exactly and what the author's intentions may have been.
    - First,"""
 
    # 解释代码的模版
    explain_code_template=PromptTemplate(
        input_variables=["unit_test_package","function_to_test"],
        template=explain_code
    )
    # llm设置
    explain_code_llm=Internlm()
    explain_code_llm.load_model(model_path)
    # 解释代码的LLMChain
    explain_code_step=LLMChain(llm=explain_code_llm,prompt=explain_code_template,output_key="code_explaination")
 
 
 
 
    # 创建测试计划示例的步骤
    test_plan = """
        
    A good unit test suite should aim to:
    - Test the function's behavior for a wide range of possible inputs
    - Test edge cases that the author may not have foreseen
    - Take advantage of the features of `{unit_test_package}` to make the tests easy to write and maintain
    - Be easy to read and understand, with clean code and descriptive names
    - Be deterministic, so that the tests always pass or fail in the same way
    `{unit_test_package}` has many convenient features that make it easy to write and maintain unit tests. We'll use them to write unit tests for the function above.
    For this particular function, we'll want our unit tests to handle the following diverse scenarios (and under each scenario, we include a few examples as sub-bullets):
    -"""
    # 测试计划的模版
    test_plan_template=PromptTemplate(
        input_variables=["unit_test_package","function_to_test","code_explaination"],
         # 解释代码需求+OpenAI给出的解释代码说说吗+本次测试计划要求的promot
        template=explain_code+"{code_explaination}"+test_plan
    )
    # llm设置
    # test_plan_llm=OpenAI(model_name='text-davinci-002',max_tokens=1000,temperature=0.4,
    #                          top_p=1,stop=["\n\n", "\n\t\n", "\n \n"])
    # test_plan_llm=ChatGLM3()
    # test_plan_llm.load_model(model_path)
    # test_plan_step=LLMChain(llm=test_plan_llm,prompt=test_plan_template,output_key="test_plan")
    test_plan_llm=Internlm()
    test_plan_llm.load_model(model_path)
    test_plan_step=LLMChain(llm=test_plan_llm,prompt=test_plan_template,output_key="test_plan")
 
 
 
 
 
    # 撰写测试代码的步骤
    starter_comment = "Below, each test case is represented by a tuple passed to the @pytest.mark.parametrize decorator"
    prompt_to_generate_the_unit_test = """
    Before going into the individual tests, let's first look at the complete suite of unit tests as a cohesive whole. We've added helpful comments to explain what each line does.
    ```python
    import {unit_test_package}  # used for our unit tests
    {function_to_test}
    #{starter_comment}"""
    
    # 单元测试的prompt----------------------------------------------------------------------
    unit_test_template=PromptTemplate(
        input_variables=["unit_test_package","function_to_test","code_explaination","test_plan","starter_comment"],
        template= explain_code + "{code_explaination}" + test_plan + "{test_plan}" + prompt_to_generate_the_unit_test
    )
    unit_test_llm=Internlm()
    unit_test_llm.load_model(model_path)
    unit_test_step = LLMChain(llm=unit_test_llm, prompt=unit_test_template, output_key="unit_test")
 
    
    # 链式调用
    sequential_chain =SequentialChain(chains=[explain_code_step, test_plan_step, unit_test_step],
                                      input_variables=["unit_test_package", "function_to_test", "starter_comment"], verbose=True)
    answer = sequential_chain.run(unit_test_package=unit_test_package, function_to_test=function_to_test, starter_comment=starter_comment)
    return f"""#{starter_comment}""" + answer
 
 
# 要写单元测试的代码
code = """
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    elif minutes > 0:
        return f"{minutes}min{seconds}s"
    else:
        return f"{seconds}s"
"""
 
import ast
# 1.调用OpeenAI得到测试单元代码
# 2.将得到的代码语义检查看是否有问题，有问题重新走第一步，直到成功！如果一直有问题超过三次则不请求处理
def write_unit_test_automatically(code,model_path,retry=3):
  # 得到OpenAI给的单元测试代码
  unit_test_code=write_unit_test(code,model_path=model_path)
  # 测试代码+单元测试代码
  all_code=code+unit_test_code
  tried=0
  print(all_code)
  # 如果异常则再次请求得到代码，最多三次
  while tried < retry:
    try:
      # 语法检查
      ast.parse(all_code)
      return all_code
    except SyntaxError as e:
      print(f"Syntax error in generated code: {e}")
      all_code=code+unit_test_code
      tried+=1
 
 
# 调用方法
model_path = "/workspace/langchain-clutter/models/internlm2-chat-1_8b"
print(write_unit_test_automatically(code,model_path=model_path))