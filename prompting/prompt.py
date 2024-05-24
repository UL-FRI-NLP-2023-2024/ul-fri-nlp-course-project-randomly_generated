import dspy
import huggingface_hub
from tqdm import tqdm 
import jsonlines


# Initialize huggingface


class default_winograd(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="3 tags with content in brackets. sentence  - problem statement, option1/option2 - option that might be true for the given statement ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")

class fewshot_winograd(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="4 tags with content in brackets. sentence  - problem statement, option1/option2 - option that might be true for the given statement, hint - examples of similarly structured statements with correct answers ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")


class default_anli(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="4 tags with content in brackets. obs1/obs2  - observations describing the problem statement, hyp1/hyp2 - option that might be true for the given statement ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")


class fewshot_anli(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="5 tags with content in brackets. obs1/obs2  - observations describing the problem statement, hyp1/hyp2 - option that might be true for the given statement , hint - examples of similarly structured statements with correct answers ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")


class default_physical(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="3 tags with content in brackets. goal  - problem statement, sol1/sol2 - option that might be true for the given statement ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")

class fewshot_physical(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="4 tags with content in brackets. goal  - problem statement, sol1/sol2 - option that might be true for the given statement, hint - examples of similarly structured statements with correct answers ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")




#Pass signature to ChainOfThought module
def generate_answers_hint(file_name_read, file_name_write, generator, hints=False):
    # Read and generate hints
    final_list = []
    problem_list = []
    with jsonlines.open(file_name_read) as reader:
        cnt = 0
        for obj in reader:
            problem_list.append(obj)

    problem_list = problem_list
    for cnt in tqdm(range(len(problem_list))):
        obj = problem_list[cnt]
        # Generate hints for physical
        prompt_string = ""
        for key in obj.keys():
            if key != "id" and key != "answer" and key != "hint":
                prompt_string += '"' + key + '": {' + obj[key] + '} '
        #prompt_string +='"correct": {hyp' + str(int(answers[cnt])) + '} '
        
        if hints:
            pred = generator(question=prompt_string,hint=obj["hint"])
        else:
            pred = generator(question=prompt_string)
        # Sometimes the prediction contains multiple answers and makes up additional questions
        # Based on 20 random observations we found that the real answer is the first one
        answer = ""
        try:
            answer = pred.answer.split("---")[2].split("Answer: ")[1].replace('"', "`")
        except:
            print("no answer")
            answer = ""

        try:
            answer = answer.split("Question: ")[0]
        except:
            answer = answer

        #answer = '`' + answer + '`'
        
        obj["answer"] = answer
        final_list.append(obj)
        print(answer)
        #cnt+=1
            #print(obj.keys())
            #break

    with jsonlines.open(file_name_write, mode='w') as writer:
        for obj in final_list:
            writer.write(obj)




#generate_answers(file_name_read, file_name_write, dspy.Predict(FewShot))
if __name__ == "__main__":
    TOKEN = "token from https://huggingface.co/settings/tokens"
    huggingface_hub.login(token=TOKEN)

    file_name_read = '/d/hpc/home/i7357/data/physicalQA/dev.jsonl'
    file_name_write = './physical_hints_prediction.jsonl'
    # This sets up the language model for DSPy in this case we are using mistral 7b
    llm = dspy.HFModel(model='mistralai/Mistral-7B-Instruct-v0.2')
    dspy.settings.configure(lm=llm)

    #select signature and prompting function
    #in this scenario we are using default signature and COT with hints
    
    generate_answer_hint = dspy.ChainOfThoughtWithHint(default_physical)
    # hint is set to True if we use ChainOfThoughtWithHint
    # generate_answers_hint(file_name_read, file_name_write, generate_answer_hint, hints=True)

    ##the fewshot signature is used only for fewshot prompting, with basic dspy.Predict
    generate_answer_example = dspy.Predict(fewshot_physical)

    generate_answer_hint(file_name_read, file_name_write, generate_answer_example, hints=False)
    

  


