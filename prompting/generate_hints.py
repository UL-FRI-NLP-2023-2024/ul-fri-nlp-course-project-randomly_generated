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


#Define a simple signature for basic question answering
#"""
file1 = open("anli.lst", 'r')
answers = file1.read().split()
file1.close()
#class HintSigature(dspy.Signature):
#    """One sentence which is a hint towards the correct answer, but not the answer itself"""
#    question = dspy.InputField(desc="a problem made up of a problem sentence, which comes after 'sentence', and options that represent possible answers, the id of the correct solution is in the field 'answer'")
#    answer = dspy.OutputField(desc="one sentence which is a vague hint towards the correct answer, but not the answer itself")

# Physical
#class HintSigature(dspy.Signature):
#    """One sentence which is a hint towards the correct answer, but not the answer itself"""
#    question = dspy.InputField(desc="a problem made up of 4 tags whos contents are in brackets after the tag. The problem statement is in tag 'goal', options for answers are in tags 'sol1' and 'sol2', the correct answer is in the field 'correct'")
#    answer = dspy.OutputField(desc="one sentence which is a vague hint towards the correct answer, but not the answer itself")

# Anli
class HintSigature(dspy.Signature):
    """One sentence which is a hint towards the correct answer, but not the answer itself"""
    question = dspy.InputField(desc="a problem made up of 4 tags whos contents are in brackets after the tag. Two connected problem statements are in tags 'obs1' and 'obs2', options for correct conclusions from the two statements are in tags 'hyp1' and 'hyp2', the correct conclusion is in the field 'correct'")
    answer = dspy.OutputField(desc="one sentence which is a vague hint towards the correct answer, but not the answer itself")



#Define a simple signature for basic question answering
#class FewShot(dspy.Signature):
#    """Five examples of simmilar questions with the correct answers"""
#    question = dspy.InputField(desc="a problem made up of a problem sentence, which comes after 'sentence', and options that represent possible answers, the correct result is in the field 'answer'")
#    answer = dspy.OutputField(desc="3 examples simmilar of simmilar problems. Each example is written in the form example1: {example sentence}, answer1: {correct answer}" )

# Physical
#class FewShot(dspy.Signature):
#    """3 examples of simmilar questions with the correct answers"""
#    question = dspy.InputField(desc="a problem made up of 4 tags whos contents are in brackets after the tag. The problem statement is in tag 'goal', options for answers are in tags 'sol1' and 'sol2', the correct answer is in the field 'correct'")
#    answer = dspy.OutputField(desc="3 examples of simmilar problems. Each example is written in the form example1: {example sentence}, answer1: {correct answer}" )

# Anli
class FewShot(dspy.Signature):
    """3 examples of simmilar questions with the correct answers"""
    question = dspy.InputField(desc="a problem made up of 4 tags whos contents are in brackets after the tag. Two connected problem statements are in tags 'obs1' and 'obs2', options for correct conclusions from the two statements are in tags 'hyp1' and 'hyp2', the correct conclusion is in the field 'correct'")
    answer = dspy.OutputField(desc="3 examples of simmilar problems. Each example is written in the form example1: {example sentence}, answer1: {correct answer}" )


def generate_answers(file_name_read, file_name_write, answer_generation_function):
    # Read and generate hints
    new_prompt_list = []
    problem_list = []
    with jsonlines.open(file_name_read) as reader:
        cnt = 0
        for obj in reader:
            problem_list.append(obj)

    problem_list = problem_list[:300]
    for cnt in tqdm(range(len(problem_list))):
        obj = problem_list[cnt]

        # Generate hints for physical
        prompt_string = ""
        for key in obj.keys():
            if key != "id":
                prompt_string += '"' + key + '": {' + obj[key] + '} '
        prompt_string +='"correct": {hyp' + str(int(answers[cnt])) + '} '
        
        pred = answer_generation_function(question=prompt_string)
        # Sometimes the prediction contains multiple answers and makes up additional questions
        # Based on 20 random observations we found that the real answer is the first one
        answer = ""
        try:
            answer = pred.answer.split("---")[2].split("Answer: ")[1].replace('"', "`")
        except:
            answer = ""

        try:
            answer = answer.split("Question: ")[0]
        except:
            answer = answer

        answer = '`' + answer + '`'
        
        obj["hint"] = answer
        new_prompt_list.append(obj)
        print(obj)
        #cnt+=1
            #print(obj.keys())
            #break

    with jsonlines.open(file_name_write, mode='w') as writer:
        for obj in new_prompt_list:
            writer.write(obj)
            
    """
    file1 = open(file_name_read, 'r')
    Lines = file1.readlines()
    Lines_modified = []
    # Strips the newline character
    for line in tqdm(range(len(Lines))):
        current_line = (Lines[line].strip())[1:-1]
        prompt = '"sentence"' + current_line.split('"sentence"')[-1]
        pred = answer_generation_function(question=prompt)
        # Sometimes the prediction contains multiple answers and makes up additional questions
        # Based on 20 random observations we found that the real answer is the first one
        answer = ""
        try:
            answer = pred.answer.split("---")[2].split("Answer: ")[1].replace('"', "`")
        except:
            answer = ""

        try:
            answer = answer.split("Question: ")[0]
        except:
            answer = answer
        
        hint = ', "hint": "' + answer +'"'
        if line == 1:
            print(answer)
        #Lines_modified.append(Lines[line].strip() + "\n")

        Lines_modified.append("{" + prompt + hint + "} \n ")
        if(line<5):
            print(Lines_modified[line])        
        #pred = generate_answer(question=prompt)
    file1.close()

    file1 = open(file_name_write, 'w')
    file1.writelines(Lines_modified)
    file1.close()

    """
file_name_read = '/d/hpc/home/am21827/anli.jsonl'
file_name_write = './anli_with_hints.jsonl'
generate_answers(file_name_read, file_name_write, dspy.Predict(HintSigature))
file_name_write = './anli_with_examples.jsonl'
generate_answers(file_name_read, file_name_write, dspy.Predict(FewShot))
