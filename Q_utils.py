import re
import torch

def extract_findings(full_report):
    if 'findings:' in full_report.lower():
        ind = full_report.lower().find('findings:')
        findings_section = full_report[ind+len("findings:"):].strip()
        findings_section = findings_section.split('\n\n')[0] # take just the first paragraph
        findings_section = re.sub(r'\s+', ' ', findings_section)
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
    
def get_parameter_count(model):
    param_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{param_count:,}")
    return param_count