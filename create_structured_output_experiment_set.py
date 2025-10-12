import os
import json
import requests
import pandas as pd
from io import StringIO

# API Configuration
EVALAP_API_URL = "http://localhost:8000/v1"
EXPERIMENT_SET_NAME = "structured_output_model_comparison_v1"
DATASET_NAME = "structured_output_test_1"
ALBERT_API_KEY = os.getenv("ALBERT_API_KEY")
ALBERT_API_URL = "https://albert.api.etalab.gouv.fr/v1"

# Authentication
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
if not ADMIN_TOKEN:
    print("❌ ADMIN_TOKEN environment variable is not set!")
    print("   Please set it with: export ADMIN_TOKEN='your-token'")
    exit(1)

HEADERS = {
    "Authorization": f"Bearer {ADMIN_TOKEN}",
    "Content-Type": "application/json"
}

# First, fetch the dataset to get the queries
print("Fetching dataset to prepare experiments...")
try:
    response = requests.get(
        f"{EVALAP_API_URL}/dataset",
        params={"name": DATASET_NAME, "with_df": True},
        headers=HEADERS
    )
    if response.status_code == 200:
        dataset_data = response.json()
        df = pd.read_json(StringIO(dataset_data["df"]))
        print(f"✅ Loaded dataset with {len(df)} rows")
    else:
        print(f"❌ Failed to fetch dataset: {response.status_code}")
        print(response.text)
        exit(1)
except Exception as e:
    print(f"❌ Error fetching dataset: {e}")
    exit(1)

# Define the JSON schema for structured extraction
with open("./notebooks/mdso-dataset/mdso-admin-schema.json", "r") as f:
    json_schema = json.loads(f.read())

# Define prompt variations for extraction
prompt_templates = {
#     "detailed": f"""Extract structured information from the text below.

# You MUST return valid JSON matching this exact schema:
# {json.dumps(json_schema, indent=2)}

# Instructions:
# - Carefully read the text and identify all entities
# - Return ONLY the requested entities in the specified format
# - Do not add explanations or markdown formatting
# - Ensure the JSON is valid and parseable

# Text to analyze:
# {{{{text}}}}

# Return only valid JSON:""",
    
    "simple": f"""Extract entities from this text and return as JSON matching this schema:
{json.dumps(json_schema, indent=2)}

# Text: {{{{text}}}}

# JSON output:""",
    "no_guide": "extract content"
}

# Define system prompt variations
system_prompts = {
    # "technical": "You are a precise JSON extraction system. Always return valid, parseable JSON without any additional text or formatting.",
    # "assistant": "You are a helpful assistant specialized in extracting structured information from text.",
    "base": "Extract informations from the text according to the provided JSON schema.",
    "none": None  # No system prompt
}

# Temperature variations
temperatures = [0.0, 0.7]
# temperatures = [0.0, 0.3, 0.7]

# Models to compare for extraction - configurable with dict structure
# Each model should have: name, base_url, api_key
models_to_test = [
    # {
    #     "name": "albert-small",
    #     "base_url": ALBERT_API_URL,
    #     "api_key": ALBERT_API_KEY
    # },
    {
        "name": "moonshotai/kimi-k2-instruct",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY")
    },
    {
        "name": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY")
    }
]

# Build the experiment set configuration
expset_name = "structured_output_model_comparison_v1"

# Generate dynamic readme based on configured models
model_names = [m["name"] for m in models_to_test]
expset_readme = f"""Comparing {', '.join(model_names)} for structured output extraction capabilities.
Testing variations in:
- Models: {', '.join(model_names)} (as extraction models in 'model' field)
- Extraction prompts: {', '.join(prompt_templates.keys())}
- System prompts: {', '.join(system_prompts.keys())}
- Temperatures: {', '.join(str(t) for t in temperatures)}

The 'model' field contains the model being evaluated for extraction.
The 'judge_model' evaluates the quality of extracted JSON against ground truth.

Model configurations:
""" + '\n'.join([f"- {m['name']}: {m['base_url']}" for m in models_to_test])

# Fixed judge model for evaluation (can be any reliable model)
# judge_model_config = {
#     "name": "albert-small",  # Using albert-large as judge for consistency
#     "base_url": ALBERT_API_URL,
#     "api_key": ALBERT_API_KEY,
#     "sampling_params": {"temperature": 0.0},  # Low temp for consistent judging
#     "aliased_name": "judge_albert_small"
# }

# Create experiments for each model and configuration combination
experiments = []
experiment_counter = 0

for model_info in models_to_test:
    # Extract model configuration from dict
    model_name = model_info["name"]
    model_base_url = model_info["base_url"]
    model_api_key = model_info["api_key"]
    
    for prompt_name, prompt_template in prompt_templates.items():
        for sys_prompt_name, sys_prompt in system_prompts.items():
            for temp in temperatures:
                # Configure the MODEL being evaluated (does the extraction)
                # We need to format the prompt with the query field
                formatted_prompt = prompt_template.replace("{{text}}", "{query}")
                
                model_config = {
                    "name": model_name,
                    "base_url": model_base_url,
                    "api_key": model_api_key,
                    "system_prompt": sys_prompt,
                    "prelude_prompt": formatted_prompt,  # This will use the query field
                    "sampling_params": {
                        "temperature": temp,
                        # "max_tokens": 500,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "entity_extraction",
                                "schema": json_schema,
                                # "strict": True # not working for my tests with groq
                            }
                        }
                    },
                    "aliased_name": f"{model_name}_prompt-{prompt_name}_sys-{sys_prompt_name}_temp-{temp}"
                }
                
                # Create the experiment
                experiment = {
                    "name": f"{expset_name}__{experiment_counter}",
                    "dataset": DATASET_NAME,
                    "model": model_config,  # This model does the extraction
                    # "judge_model": judge_model_config,  # This evaluates the extraction
                    "metrics": ["llm_structured_output"],
                    "with_vision": False,
                }
                
                experiments.append(experiment)
                experiment_counter += 1

# Create the experiment set configuration
expset = {
    "name": expset_name,
    "readme": expset_readme,
    "experiments": experiments
}

# Submit the experiment set
print("\nCreating experiment set...")
print(f"  Models to compare: {[m['name'] for m in models_to_test]}")
print(f"  Prompt templates: {list(prompt_templates.keys())}")
print(f"  System prompts: {list(system_prompts.keys())}")
print(f"  Temperatures: {temperatures}")
print(f"  Total experiments: {len(experiments)}")

# Calculate experiments per model
experiments_per_model = len(prompt_templates) * len(system_prompts) * len(temperatures)
for model in models_to_test:
    print(f"    - {model['name']}: {experiments_per_model} experiments")

try:
    response = requests.post(f'{EVALAP_API_URL}/experiment_set', json=expset, headers=HEADERS, timeout=60)
    
    if response.status_code == 200:
        try:
            resp = response.json()
            if "id" in resp:
                expset_id = resp["id"]
                print(f'\n✅ Created experiment set: {resp["name"]} (ID: {resp["id"]})')
                print(f"   Status: {resp.get('status', 'unknown')}")
                print(f"   Experiments: {len(resp.get('experiments', []))}")
                print(f"\n📊 Experiment design:")
                print(f"   • MODEL field: {[m['name'] for m in models_to_test]} perform extraction")
                print(f"   • These models extract entities from the 'query' field")
                print(f"   • Extracted JSON goes to 'answer' column")
                print(f"   • Results show which model extracts entities better")
                print(f"\n💡 Correct data flow:")
                print(f"   query → MODEL (extracts) → answer → JUDGE_MODEL (scores) → metric score")
            else:
                print(f"❌ Unexpected response format: {resp}")
                exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"   Raw response: '{response.text}'")
            exit(1)
    else:
        print(f"❌ API request failed with status {response.status_code}")
        print(f"   Response: '{response.text}'")
        exit(1)

except requests.exceptions.ConnectionError:
    print(f"❌ Cannot connect to Evalap API at {EVALAP_API_URL}")
    print("   Make sure the Evalap server is running with:")
    print("   uvicorn evalap.api.main:app --reload --host 0.0.0.0 --port 8000")
    exit(1)
except requests.exceptions.RequestException as e:
    print(f"❌ Request error: {e}")
    exit(1)