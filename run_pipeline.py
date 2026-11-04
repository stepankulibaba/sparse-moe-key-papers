"""Rebuild frozen analysis, figures and checks without network access.

Usage: python -X utf8 run_pipeline.py [--skip-stress] [--no-figures]
Install dependencies with: python -m pip install -r requirements.txt
Collection and full-text extraction are separate, opt-in network operations.
PDF source files are not distributed; their URLs and hashes are retained in
data/source_pdf_manifest.json. Rebuilding uses saved bibliography/context text.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-stress',action='store_true',help='Skip 2,000 perturbations and their saved-artifact checks')
    parser.add_argument('--no-figures',action='store_true',help='Skip PNG/SVG figure exports')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    (root/'output').mkdir(exist_ok=True)
    scripts=['analyze.py','custom_metrics.py','semantic_metrics.py','test_custom_metrics.py','verify.py','verify_extensions.py']
    if not args.skip_stress:scripts+=['robustness.py','verify_robustness.py']
    if not args.no_figures:
        scripts+=['figures.py','custom_figures.py']
        if not args.skip_stress:scripts+=['robustness_figures.py']
    for script in scripts:
        print('Running',script,flush=True)
        subprocess.run([sys.executable,'-X','utf8',str(root/script)],cwd=root,check=True)
    print('Offline analysis complete. Results and verification records: output/')


if __name__=='__main__':main()
