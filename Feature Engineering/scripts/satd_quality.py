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
def analyze_satd_quality(comment: str) -> int:
    if not comment or comment.strip() == "":
        return 1
    """
    Using LLM scoring to obtain the quality score of satd annotations
    """
    dimension_judge_prompt_body = (
        "Your input is a comment acknowledging that there is a technical debt in the relevant code. "
        "You need to carefully understand the comment and ultimately provide a score. "
        "If a comment makes it unclear what problem it is describing, "
        "or if it is simply complaining without any analysis or suggestions for improvement, "
        "or if it is merely describing or explaining the functionality of the code, then you should output '<1>'."
        "If a comment points out a problem in the code, "
        "or if it raises a question about the current code or operation, "
        "or if it suggests a to-do item without providing a specific implementation method, "
        "or if it directly includes a bug number, URL, or issue number, we consider it as describing an issue, and you should output '<2>'."
        "If a comment provides a specific action suggestion or solution, then regardless of whether it explicitly mentions a problem in the code, you should output '<3>'."
        "Additionally, if the comment is split into multiple lines, each describing different issues or fixes, "
        "you should focus on the sentence led by specific keywords such as 'todo' or 'fixme', and score based on that sentence.\n\n"
    )
    dimension_judge_prompt_examples = '''
Example1:
Comment:
Hack: I use jxpath to populate the context object's properties
in the jexl context
Response:<1>

Example2:
Comment:
Ensure that the 'page=xyz' attribute is removed
FIXME: Is it really the mandate of this routine to
do that?
Response:<1>

Example3:
Comment:
Oooo! This is really bad...
Response:<1>

Example4:
Comment:
FIXME: Not yet implemented
Response:<2>

Example5:
Comment:
workaround Bug 51939
Response:<2>

Example6:
todo: Do we want to give the user control over which types have priority?
Response:<2>

Example7:
Comment:
Sends more notifications if haven't received enough. Otherwise
processes new notification.
Response:<3>

Example8:
Comment:
FIXME: In case of exceptions should absolutely
remove the uploaded file.
Response:<3>

Example9:
Comment:
TODO Needs constant on SubsystemConstants.
Response:<3>
'''
    prompt = dimension_judge_prompt_body + dimension_judge_prompt_examples + f"\nComment:\n{comment}\nResponse:"

    # Call OpenAI API
    response = client.chat.completions.create(
        model="deepseek-v3",  # Model Name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    score = extract(response.choices[0].message.content.strip())
    print(score)
    
    if score not in range(1,4):
        print(f"[✖ ]Score {score} out of range, set to 0")
        score = 1
        
    return score

def extract(response: str) -> int:
    matches = re.findall(r'<([123])>', response)
    return int(matches[-1]) if matches else 1


def process_raw_data(input_path, output_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    features = []
    for record in raw_data:
        print('-' * 60)
        satd_id = record.get('satd_id')
        satd_comment = record.get('f_comment')
            
        satd_quality_score = analyze_satd_quality(satd_comment)
            
        features.append({
            "satd_id": satd_id,
            "satd_quality_score": satd_quality_score
        })
        print(f"satd_id: {satd_id} \nsatd_comment: {satd_comment}")


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
    