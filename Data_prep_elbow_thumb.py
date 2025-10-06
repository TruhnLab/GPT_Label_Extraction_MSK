import os
import json
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split
import re


def prepare_data(image_dir, label_dir, output_csv):
    patient_ids = []
    labels = []
    label_keys = []

    # Iterate over label files to determine the structure
    for txt_file in os.listdir(label_dir):
        if txt_file.endswith('.txt'):
            with open(os.path.join(label_dir, txt_file), 'r') as f:
                label_data = json.load(f)
            label_keys = list(label_data.keys())
            break

    # Iterate over label files to collect patient IDs and labels
    for txt_file in os.listdir(label_dir):
        if txt_file.endswith('.txt'):
            patient_id = os.path.splitext(txt_file)[0]
            label_path = os.path.join(label_dir, txt_file)
            ap_image_path = os.path.join(image_dir, patient_id, 'ap.png')
            lat_image_path = os.path.join(image_dir, patient_id, 'lat.png')

            if os.path.exists(ap_image_path) and os.path.exists(lat_image_path):
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                patient_labels = []
                skip_patient = False
                for key in label_keys:
                    if key in label_data:
                  
                        finding = label_data[key]
                        if isinstance(finding, dict):
                            patient_labels.append(1 if finding["finding"] == True else 0)
                        elif isinstance(finding, bool):
                            patient_labels.append(1 if finding else 0)
                    else:
                        skip_patient = True
                        print(f"Key {key} not found in {txt_file} for patient {patient_id}, skipping...")
                        break
                if not skip_patient:
                    patient_ids.append(patient_id)
                    labels.append(patient_labels)

    # Create a DataFrame to count the number of members for each class
    df_labels = pd.DataFrame(labels, columns=label_keys)
    print(df_labels.shape[0])

    # Prepare data for iterative_train_test_split
    X = pd.DataFrame({
        "AP_Image_Path": [os.path.join(image_dir, patient_id, 'ap.png').replace("\\", "/") for patient_id in patient_ids],
        "LAT_Image_Path": [os.path.join(image_dir, patient_id, 'lat.png').replace("\\", "/") for patient_id in patient_ids],
    })
    y = df_labels

    # Split data into train+val and test sets
    X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(X.values, y.values, test_size=0.2)

    # Further split train+val into train and val sets
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_train_val, y_train_val, test_size=0.2)

    # Combine data into DataFrames
    train_df = pd.DataFrame(X_train, columns=["AP_Image_Path", "LAT_Image_Path"])
    val_df = pd.DataFrame(X_val, columns=["AP_Image_Path", "LAT_Image_Path"])
    test_df = pd.DataFrame(X_test, columns=["AP_Image_Path", "LAT_Image_Path"])

    train_df[label_keys] = y_train
    val_df[label_keys] = y_val
    test_df[label_keys] = y_test

    train_df['Phase'] = 'train'
    val_df['Phase'] = 'val'
    test_df['Phase'] = 'test'

    # Combine all DataFrames
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Filter classes that appear at least once in each phase and print the occurrences
    for label in label_keys:
        train_occurrence = final_df[final_df['Phase'] == 'train'][label].sum()
        val_occurrence = final_df[final_df['Phase'] == 'val'][label].sum()
        test_occurrence = final_df[final_df['Phase'] == 'test'][label].sum()

        if train_occurrence < 1 or val_occurrence < 1 or test_occurrence < 1:
            total_occurrence = train_occurrence + val_occurrence + test_occurrence
            print(f"Discarded Class: {label} with occurrences - Train: {train_occurrence}, Val: {val_occurrence}, Test: {test_occurrence}, Total: {total_occurrence}")
            final_df = final_df.drop(columns=[label])	    

    # Save the final DataFrame to CSV
    final_df.to_csv(output_csv, index=False)

    # Print the count of each label in each phase
    print("Label counts in each phase:")
    for phase in ['train', 'val', 'test']:
        phase_df = final_df[final_df['Phase'] == phase]
        print(f"\n{phase} phase:")
        for label in final_df.columns[2:-1]:  # Skip image paths and phase columns
            count = phase_df[label].sum()
            print(f"{label}: {count}")

    return final_df

def main():
    image_dir = 'Images'
    label_dir = 'Output/Labels'
    output_csv = 'Output/Elbow.csv'
    
   
    final_df = prepare_data(image_dir, label_dir, output_csv)

if __name__ == "__main__":
    main()