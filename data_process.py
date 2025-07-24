from datasets import load_dataset

def preprocess_truthfulqa_for_reft(examples):
    TRUTHFULQA_SUBSPACE = [0,1,2,3,4,5]
    questions = []
    outputs = []
    subspaces = []
    for i in range(len(examples["Question"])):
        question = examples["Question"][i]
        questions.append(question)
        best_answer = examples["Best Answer"][i]
        outputs.append(best_answer)
        subspaces.append(TRUTHFULQA_SUBSPACE)
        for j in range(examples["Correct Answers"][i].count(";")+1):
            correct_answers = examples["Correct Answers"][i].split(';')[j]
            questions.append(question)
            outputs.append(correct_answers)
            subspaces.append(TRUTHFULQA_SUBSPACE)
    return {"instruction": questions, "output": outputs, "subspaces": subspaces}

def preprocess_bbq_for_reft(examples):
    
    BBQ_SUBSPACE = [0,1,2,3,6,7]
    label = examples["answer"]
    output = examples["choices"][label]
    examples["instruction"] = examples["context"]+examples["question"]
    examples["output"] = output
    examples["subspaces"] = BBQ_SUBSPACE
    #print(examples["subspaces"])
    
    return examples

def preprocess_refusal_for_reft(examples):
    SUBSPACE = [0,1,5,6,7]
    examples["instruction"] = examples["input"]
    examples["output"] = examples["output"]
    examples["subspaces"] = SUBSPACE
    return examples

def preprocess_alpaca_for_reft(examples):
    SUBSPACE = [0,1,2,3,4]
    instruction = examples['instruction']
    inputs = examples['input']
    output = examples['output']
    examples["instruction"] = instruction + inputs
    examples["output"] = output
    examples["subspaces"] = SUBSPACE
    return examples

def preprocess_helpsteer_for_reft1(examples):
    HELP_SUBSPACE1 = [0,1,2,3]
    examples["instruction"] = examples["prompt"]
    examples["output"] = examples["response"]
    examples["subspaces"] = HELP_SUBSPACE1
    return examples

def preprocess_helpsteer_for_reft2(examples):
    HELP_SUBSPACE2 = [0,1,4,5]
    examples["instruction"] = examples["prompt"]
    examples["output"] = examples["response"]
    examples["subspaces"] = HELP_SUBSPACE2
    return examples

def preprocess_helpsteer_for_reft3(examples):
    HELP_SUBSPACE3 = [0,1,6,7]
    examples["instruction"] = examples["prompt"]
    examples["output"] = examples["response"]
    examples["subspaces"] = HELP_SUBSPACE3
    return examples
