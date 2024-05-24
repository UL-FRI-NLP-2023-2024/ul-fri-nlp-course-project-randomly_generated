import dspy
import huggingface_hub
from tqdm import tqdm 
import jsonlines

# Initialize huggingface
TOKEN = "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"
huggingface_hub.login(token=TOKEN)

# Use model mistralai/Mistral-7B-Instruct-v0.2
# This sets up the language model for DSPy in this case we are using mistral 7b
llm = dspy.HFModel(model='mistralai/Mistral-7B-Instruct-v0.2')
# This sets the language model for DSPy.
dspy.settings.configure(lm=llm)


class Default_winograd(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="3 tags with content in brackets. sentence  - problem statement, option1/option2 - option that might be true for the given statement ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")

class FewShot_winograd(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="4 tags with content in brackets. sentence  - problem statement, option1/option2 - option that might be true for the given statement, hint - examples of similarly structured statements with correct answers ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")


class Default_anli(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="4 tags with content in brackets. obs1/obs2  - observations describing the problem statement, hyp1/hyp2 - option that might be true for the given statement ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")


class FewShot_anli(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="5 tags with content in brackets. obs1/obs2  - observations describing the problem statement, hyp1/hyp2 - option that might be true for the given statement , hint - examples of similarly structured statements with correct answers ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")


class Default_physical(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="3 tags with content in brackets. goal  - problem statement, sol1/sol2 - option that might be true for the given statement ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")

class FewShot_physical(dspy.Signature):
    """1 for option 1 2 for option 2"""
    question = dspy.InputField(desc="4 tags with content in brackets. goal  - problem statement, sol1/sol2 - option that might be true for the given statement, hint - examples of similarly structured statements with correct answers ")
    answer = dspy.OutputField(desc="1 if option 1 text is true, 2 if option 2 text is true, only one can be true at a time, output a single integer")




#Pass signature to ChainOfThought module
generate_answer_hint = dspy.ChainOfThoughtWithHint(CoT_hint_winograd)
#Pass signature to ChainOfThought module
generate_answer_example = dspy.Predict(FewShot_example_winograd)

def generate_answers(file_name_read, file_name_write, generate_answer_hint):
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
        
        pred = generate_answer_hint(question=prompt_string,hint=obj["hint"])
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


file_name_read = '/d/hpc/home/am21827/d_physical/physical_hint.jsonl'
file_name_write = './physical_hints_prediction.jsonl'
generate_answers(file_name_read, file_name_write, generate_answer_hint)
file_name_write = './anli_with_examples.jsonl'
#generate_answers(file_name_read, file_name_write, dspy.Predict(FewShot))
            