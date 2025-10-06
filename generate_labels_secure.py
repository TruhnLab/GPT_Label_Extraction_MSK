import os
import json
import pandas as pd
from  openai import AzureOpenAI
from datetime import timedelta
import time
from dotenv import load_dotenv

load_dotenv(override=True)
# Load credentials from environment variables for security

# Function to generate labels for the reports using Azure OpenAI
def generate_labels(finding, template_json, client):
    messages = [
       {"role": "system", "content": "You are a helpful assistant that labels medical findings based on a given template and outputs JSON"},
{"role": "user", "content": f"""
Fill out the following template {template_json} in JSON format according to the information given in {finding}. Adhere strictly to the structure of the template. Only fill out the "finding" fields and do not add anything else.
1. Mark a finding as true only if it is explicitly confirmed in the report with no uncertainty or hedging terms.
2. Mark a finding as uncertain if the report contains hedging or uncertainty terms like "not clearly delineable" (German: "nicht sicher abgrenzbar"), "most likely" ("am ehesten"), "possibly" ("möglicherweise"), "cannot be ruled out" ("nicht ausgeschlossen"), "suspected" ("Verdacht auf"), "a potential differential diagnosis may be X" ("als DD käme in Frage X" or "DD X") or similar phrases indicating doubt.
3. Mark a finding as false if the report explicitly states the absence of a finding or if the report contains no information related to the specific label.
4. If a specific sub-category finding (e.g., "Lateral Third Fracture") is marked as true or uncertain, ensure that the broader category (e.g., "Fracture [All Locations]") is marked accordingly.
5. Ensure no findings from other anatomic regions are considered when marking the "finding" fields. Each finding must be strictly limited to the anatomic region of interest (e.g., clavicle) specified in the template.
"""}    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        response_format= { "type": "json_object" }
    )
    
    tokens_used = response.usage.total_tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    labels = response.choices[0].message.content
    return labels, tokens_used, input_tokens, output_tokens

def main():
    # Initialize the client
    client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-03-01-preview"
    )
    
    # Load the template JSON
    template_path = "Templates/thumb.json"
    with open(template_path, "r") as file:
        template_json = file.read()

    # Load the reports
    findings_df = pd.read_csv("Mock_reports_thumb.csv")

    # Convert all PatientID values to strings and strip any leading or trailing spaces
    findings_df['ID'] = findings_df['ID'].astype(str).str.strip()

    # Initialize total token counts
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0

    results = []
    output_directory_reports = "Output/Reports"
    os.makedirs(output_directory_reports, exist_ok=True)
    output_directory_labels = "Output/Labels"
    os.makedirs(output_directory_labels, exist_ok=True)

    for index, row in findings_df.iterrows():
        patient_id = row['ID']
        finding = row['Report']
        labels_txt_filepath = os.path.join(output_directory_labels, f"{patient_id}.txt")
        if os.path.exists(labels_txt_filepath):
            print(f"Labels for PatientID {patient_id} already exist. Skipping...")
            continue
        labels, tokens_used, input_tokens, output_tokens = generate_labels(finding, template_json, client)
        labels = labels.replace("\n", "").replace("\r", "").strip()
        labels = json.loads(labels)
        total_tokens += tokens_used
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        # Save labels as a .txt file
        labels_txt_filepath = os.path.join(output_directory_labels, f"{patient_id}.txt")
        with open(labels_txt_filepath, 'w', encoding='utf-8') as labels_txt_file:
            json.dump(labels, labels_txt_file, indent=4, ensure_ascii=False)
        
        # Save report as a .txt file
        report_txt_filepath = os.path.join(output_directory_reports, f"{patient_id}.txt")
        with open(report_txt_filepath, 'w', encoding='utf-8') as report_txt_file:
            report_txt_file.write(finding)
        
        results.append({
            "PatientID": patient_id,
            "Finding": finding,
            "Labels": labels
        })

if __name__ == "__main__":
    main()
