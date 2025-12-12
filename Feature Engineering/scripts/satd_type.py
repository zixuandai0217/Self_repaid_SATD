from openai import OpenAI
import json
import os
import re
import sys

# Set API environment variables and OpenAI client
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"]
)

# Prompt and model invocation
def analyze_satd_type(comment: str) -> str:
    if not comment or comment.strip() == "":
        return "other type"
    """
    Using the LLM interface to classify SATD comments and return the classification results.
    """
    satd_judge_prompt_body = (
        "Your input is a comment acknowledging that there is a technical debt in the relevant code. "
        "You need to carefully understand the comment and classify it into one of the following SATD types. "
        "Below are the five SATD types with their definitions:\n\n"
        "1. <design debt>: These comments indicate a problem with the design of the code, such as misplaced code, lack of abstraction, long methods, poor implementation, workarounds, or temporary solutions.\n"
        "2. <defect debt>: These comments indicate that a part of the code does not have the expected behavior, meaning there is a defect in the code.\n"
        "3. <documentation debt>: These comments express that there is no proper documentation supporting that part of the program.\n"
        "4. <requirement debt>: These comments express incompleteness of the method, class, or program.\n"
        "5. <test debt>: These comments express the need for implementation or improvement of the current tests.\n\n"
        "Please analyze the comment and output '<SATD type>'."
        "If the comment suggests multiple types, choose the most prominent one based on the primary intent.\n\n"
    )
    satd_judge_prompt_examples = '''
Example 1:
Comment:
TODO: - This method is too complex, lets break it up
Response: <design debt>

Example 2:
Comment:
TODO: really should be a separate class
Response: <design debt>

Example 3:
Comment:
Bug in above method
Response: <defect debt>

Example 4:
Comment:
WARNING: the OutputStream version of this doesn’t work!
Response: <defect debt>

Example 5:
Comment:
FIXME This function needs documentation.
Response: <documentation debt>

Example 6:
Comment:
TODO Document the reason for this
Response: <documentation debt>

Example 7:
Comment:
TODO no methods yet for getClassname.
Response: <requirement debt>

Example 8:
Comment:
TODO: The copy function is not yet * completely  implemented - so we will * have some exceptions here and there.
Response: <requirement debt>

Example 9:
Comment:
TODO - need a lot more tests
Response: <test debt>

Example 10:
Comment:
TODO enable some proper tests!!
Response: <test debt>
'''
    prompt = satd_judge_prompt_body + satd_judge_prompt_examples + f"\nComment:\n{comment}\nResponse:"

    # Call LLM API
    response = client.chat.completions.create(
        model="deepseek-v3",  # Model Name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    score = response.choices[0].message.content.strip()
    return extract(score)

def extract(response: str) -> str:
    allowed_types = {'design debt', 'defect debt', 'documentation debt', 'requirement debt', 'test debt'}
    matches = re.findall(r'<([^>]+)>', response)
    if matches:
        last_type = matches[-1].strip()
        return last_type if last_type in allowed_types else 'defect debt'
    return 'defect debt'


def process_raw_data(input_path, output_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    features = []
    for record in raw_data:
        print("-" * 60)
        satd_id = record.get('satd_id')
        satd_comment = record.get('f_comment')    
        
        satd_type = analyze_satd_type(satd_comment)

        features.append({
            "satd_id": satd_id,
            "satd_type": satd_type
        })
        print(f"Processing record {satd_id}:\n{satd_comment}\n --> {satd_type}")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join(output_dir, f"{script_name}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, indent=4, ensure_ascii=False)
    print(f"Feature file successfully generated: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.normpath(
        os.path.join(current_dir, "../../Dataset/data/raw_data_final_40501.json")
    )
    FEATURES_DIR = os.path.normpath(
        os.path.join(current_dir, "../features/")
    )
    
    process_raw_data(RAW_DATA_PATH, FEATURES_DIR)
    