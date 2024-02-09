#%%
import os
import re
import csv
import pandas as pd
from tqdm import tqdm
from pprint import pprint

def extract_findings(full_report):
    if 'findings:' in full_report.lower():
        ind = full_report.lower().find('findings:')
        findings_section = full_report[ind+len("findings:"):].strip()
        findings_section = findings_section.split('\n\n')[0] # take just the first paragraph
        findings_section = re.sub(r'\s+', ' ', findings_section)
        
        # Remove IMPRESSION part that is included in the FINDINGS part
        if 'impression:' in findings_section.lower():
            ind = findings_section.lower().find('impression:')
            findings_section = findings_section[:ind].strip()
        
        return findings_section
    else:
        return ""

def extract_impression(full_report):
    if 'impression:' in full_report.lower():
        ind = full_report.lower().find('impression:')
        impression_section = full_report[ind+len("impression:"):].strip()
        impression_section = impression_section.split('\n\n')[0] # take just the first paragraph
        impression_section = re.sub(r'\s+', ' ', impression_section)
        return impression_section
    else:
        return ""

# %%
mimic_root = '/media/wonjun/HDD8TB/mimic-cxr-jpg-resized512'
split_path = os.path.join(mimic_root, 'mimic-cxr-2.0.0-split.csv')
split_df = pd.read_csv(split_path)
train_df = split_df[split_df['split']=='train']
display(train_df)

#%%
snippets_list = []

for i, row in tqdm(train_df.iterrows(), total=len(train_df), colour='magenta'):
    row = split_df.iloc[i]
    subject_id = row['subject_id']
    study_id = row['study_id']

    txt_path = os.path.join(mimic_root, 'reports', 'files',
                            f"p{str(int(subject_id))[:2]}",
                            f"p{str(int(subject_id))}",
                            f"s{str(int(study_id))}.txt")
    with open(txt_path, 'r') as handle:
        report = handle.read().strip()

    findings = extract_findings(report)
    impression = extract_impression(report)
    report = findings+impression
    sentences = re.split(r'(?<!\d)\.(?!\d)', report) # split into sentences (based on period; but avoid splitting periods used as decimals e.g. 5.5-6cm)
    sentences = [s.strip().lower() for s in sentences if len(s)>5]
    # remove sentences irrelevant to image understanding
    remove_these = [
        'previous',
        'compar',
        '__',
        'change',
        'clinical correlation',
        'recommend'
        ]
    sentences = [s for s in sentences if not any(substring in s for substring in remove_these)] 
    sentences = [s for s in sentences if not re.search(r'\b(view|ap|pa|ct|dr)\b', s, flags=re.IGNORECASE)]
    rows = [[subject_id, study_id, sentence] for sentence in sentences]
    snippets_list.extend(rows)

csv_file = 'mimic_report_snippets.csv'
with open(csv_file, 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerows(snippets_list)

#%%