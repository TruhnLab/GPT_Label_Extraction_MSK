"""
Test version of generate_labels for thumb data that creates mock labels without API calls.
Use this for testing the thumb pipeline without requiring Azure OpenAI credentials.
"""
import os
import json
import pandas as pd

def create_mock_thumb_labels_from_template(template_json, finding_text):
    """Create mock thumb labels based on simple text analysis"""
    template = json.loads(template_json)
    
    # Simple rule-based mock labeling for testing
    finding_lower = finding_text.lower()
    
    # Reset all findings to False first
    for key in template:
        template[key]["finding"] = False
    
    # Simple pattern matching for thumb-specific findings
    if "fraktur" in finding_lower or "fracture" in finding_lower:
        template["Fracture (All Locations)"]["finding"] = True
        
        # Specific bone fractures
        if "mittelhandknochen" in finding_lower or "metacarpal" in finding_lower:
            template["First Metacarpal Bone Fracture"]["finding"] = True
        elif "grundphalanx" in finding_lower or "proximal phalanx" in finding_lower:
            template["Proximal Phalanx Fracture"]["finding"] = True
        elif "endphalanx" in finding_lower or "distal phalanx" in finding_lower:
            template["Distal Phalanx Fracture"]["finding"] = True
    
    # Comminuted/fragmented fractures
    if "trümmer" in finding_lower or "kompliziert" in finding_lower or "fragmented" in finding_lower or "comminuted" in finding_lower:
        template["Comminuted or Fragmented Fracture (All Locations)"]["finding"] = True
        
        # Specific comminuted fractures
        if "mittelhandknochen" in finding_lower or "metacarpal" in finding_lower:
            template["First Metacarpal Bone - Comminuted or Fragmented Fracture"]["finding"] = True
        elif "grundphalanx" in finding_lower or "proximal phalanx" in finding_lower:
            template["Proximal Phalanx - Comminuted or Fragmented Fracture"]["finding"] = True
        elif "endphalanx" in finding_lower or "distal phalanx" in finding_lower:
            template["Distal Phalanx - Comminuted or Fragmented Fracture"]["finding"] = True
    
    # Joint problems
    if "luxation" in finding_lower or "dislocation" in finding_lower:
        template["Joint Dislocation (All Locations)"]["finding"] = True
        
        if "karpometakarpal" in finding_lower or "carpometacarpal" in finding_lower:
            template["Carpometacarpal Joint - Dislocation"]["finding"] = True
        elif "metakarpophalangeal" in finding_lower or "metacarpophalangeal" in finding_lower:
            template["Metacarpophalangeal Joint - Dislocation"]["finding"] = True
        elif "interphalangeal" in finding_lower:
            template["Interphalangeal Joint - Dislocation"]["finding"] = True
    
    # Subluxation
    if "subluxation" in finding_lower:
        template["Joint Subluxation (All Locations)"]["finding"] = True
    
    # Soft tissue changes
    if "schwellung" in finding_lower or "swelling" in finding_lower or "dactylitis" in finding_lower:
        template["Swelling/Dactylitis"]["finding"] = True
    
    # Foreign bodies
    if "fremdkörper" in finding_lower or "foreign" in finding_lower:
        template["Foreign Bodies"]["finding"] = True
    
    return template

def main():
    print("Running mock thumb label generation (no API calls)...")
    
    # Load the template JSON
    template_path = "Templates/thumb.json"
    with open(template_path, "r") as file:
        template_json = file.read()

    # Load the reports
    findings_df = pd.read_csv("Mock_reports_thumb.csv")

    # Convert all PatientID values to strings and strip any leading or trailing spaces
    findings_df['ID'] = findings_df['ID'].astype(str).str.strip()

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
            
        # Create mock labels using simple pattern matching
        labels = create_mock_thumb_labels_from_template(template_json, finding)
        
        # Save labels as a .txt file
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
        
        print(f"Created mock thumb labels for patient {patient_id}")

    print(f"Mock thumb label generation complete! Created {len(results)} label files.")

if __name__ == "__main__":
    main()
