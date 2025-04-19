#!/usr/bin/env python3
"""
Mail Image Re-scanner for Desk-Mail-Bot

This script uses Gemini to re-scan all images in mail_data/images and compare 
the results with what's in mail_log.csv to ensure accuracy.
"""

import os
import csv
import pandas as pd
import re
import cv2
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import the necessary functions from the vision script
from desk_mail_vision import (
    analyze_multipage_mail, 
    encode_image, 
    MODELS
)

# Constants
IMAGES_DIR = Path("mail_data/images")
LOG_FILE = Path("mail_data/mail_log.csv")
console = Console()

def extract_doc_id(filename):
    """Extract document ID from filename"""
    # Filenames are like: 20250419_095338_AMS_page1.jpg
    # Document ID is: 20250419_095338_AMS
    match = re.match(r'(\d{8}_\d{6}_[A-Z]+)_page\d+\.jpg', filename)
    if match:
        return match.group(1)
    # Handle error files which might have different format
    match = re.match(r'(\d{8}_\d{6}_[A-Z]+).*\.jpg', filename)
    if match:
        return match.group(1)
    # Handle even more generic patterns
    match = re.match(r'(\d{8}_\d{6}).*\.jpg', filename)
    if match:
        return match.group(1)
    return None

def load_mail_log():
    """Load the mail log CSV into a pandas DataFrame"""
    if not LOG_FILE.exists():
        console.print(f"[bold red]Mail log file not found: {LOG_FILE}[/]")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(LOG_FILE)
        console.print(f"[green]Loaded mail log with {len(df)} entries[/]")
        return df
    except Exception as e:
        console.print(f"[bold red]Error loading mail log: {str(e)}[/]")
        return pd.DataFrame()

def scan_image_directory():
    """Scan the images directory and group images by document ID"""
    if not IMAGES_DIR.exists():
        console.print(f"[bold red]Images directory not found: {IMAGES_DIR}[/]")
        return {}
    
    doc_images = defaultdict(list)
    total_images = 0
    
    # Walk through all files in the images directory
    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')) or filename == "README.md":
            continue
        
        doc_id = extract_doc_id(filename)
        if doc_id:
            full_path = IMAGES_DIR / filename
            doc_images[doc_id].append(full_path)
            total_images += 1
    
    console.print(f"[green]Found {total_images} images across {len(doc_images)} documents[/]")
    return doc_images

def load_and_process_document(doc_id, image_paths):
    """Load and process a document with its images using Gemini"""
    console.print(f"[yellow]Re-scanning document: {doc_id}[/]")
    
    # Load images and sort by page number
    images = []
    sorted_paths = sorted(image_paths, key=lambda p: 
                         int(re.search(r'page(\d+)', p.name).group(1))
                         if re.search(r'page(\d+)', p.name) else 0)
    
    for path in sorted_paths:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
    
    if not images:
        console.print(f"[bold red]Failed to load any images for document: {doc_id}[/]")
        return None
    
    # Process the document using Gemini
    with Progress() as progress:
        task = progress.add_task("[cyan]Analyzing with Gemini...", total=1)
        data = analyze_multipage_mail(images)
        progress.update(task, completed=1)
    
    if data:
        # Add document ID and image paths
        data['DocumentID'] = doc_id
        data['ImagePaths'] = ';'.join([str(p) for p in sorted_paths])
        
        # Display summary
        console.print(f"[bold green]Successfully processed document: {doc_id}[/]")
        
        return data
    else:
        console.print(f"[bold red]Failed to process document: {doc_id}[/]")
        return None

def compare_document_data(doc_id, rescan_data, mail_log_data):
    """Compare the rescanned data with what's in the mail log"""
    comparison = {}
    key_fields = ['Sender', 'DocumentDate', 'Category', 'AmountDue', 'DueDate', 'ActionNeeded']
    
    for field in key_fields:
        if field in rescan_data and field in mail_log_data:
            rescan_value = str(rescan_data[field])
            log_value = str(mail_log_data[field])
            
            # Special handling for amount due (ignore formatting)
            if field == 'AmountDue':
                # Strip everything except digits, dots, and commas
                rescan_digits = ''.join(c for c in rescan_value if c.isdigit() or c in '.,')
                log_digits = ''.join(c for c in log_value if c.isdigit() or c in '.,')
                
                # If we have numeric values, compare them
                if rescan_digits and log_digits:
                    # Try to convert to float for comparison
                    try:
                        rescan_amount = float(rescan_digits.replace(',', ''))
                        log_amount = float(log_digits.replace(',', ''))
                        
                        # Consider equal if within 1% of each other
                        if abs(rescan_amount - log_amount) < max(rescan_amount, log_amount) * 0.01:
                            comparison[field] = {
                                'match': True,
                                'rescan': rescan_value,
                                'log': log_value
                            }
                            continue
                    except ValueError:
                        pass  # Fall back to string comparison
            
            # Check if the values match exactly or if log contains rescan or vice versa
            if (rescan_value == log_value or 
                (rescan_value and rescan_value in log_value) or 
                (log_value and log_value in rescan_value)):
                comparison[field] = {
                    'match': True,
                    'rescan': rescan_value,
                    'log': log_value
                }
            else:
                comparison[field] = {
                    'match': False,
                    'rescan': rescan_value,
                    'log': log_value
                }
        else:
            comparison[field] = {
                'match': False,
                'rescan': rescan_data.get(field, "Not found"),
                'log': mail_log_data.get(field, "Not found")
            }
    
    return comparison

def display_comparison_results(comparison, doc_id):
    """Display the comparison results in a table"""
    table = Table(title=f"Document Comparison: {doc_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Mail Log", style="green")
    table.add_column("Rescan", style="yellow")
    table.add_column("Match", style="magenta")
    
    for field, data in comparison.items():
        match_str = "[green]✓[/]" if data['match'] else "[red]✗[/]"
        table.add_row(field, data['log'], data['rescan'], match_str)
    
    console.print(table)

def main():
    """Main function"""
    console.print(Panel.fit(
        "[bold green]Mail Image Re-scanner[/]\n"
        "This tool uses Gemini to re-scan all mail images and verify mail_log.csv accuracy",
        title="Desk-Mail-Bot"
    ))
    
    # Load mail log
    mail_log = load_mail_log()
    if mail_log.empty:
        console.print("[bold red]Cannot proceed without mail log data[/]")
        return
    
    # Create a dictionary of mail log data by document ID
    mail_log_dict = {}
    for _, row in mail_log.iterrows():
        doc_id = row.get('DocumentID')
        if doc_id:
            mail_log_dict[doc_id] = row.to_dict()
    
    # Scan image directory and group by document ID
    doc_images = scan_image_directory()
    if not doc_images:
        console.print("[bold red]No images found to process[/]")
        return
    
    # Compare document IDs
    mail_log_doc_ids = set(mail_log_dict.keys())
    image_doc_ids = set(doc_images.keys())
    
    # Check for documents in images but not in log
    missing_from_log = image_doc_ids - mail_log_doc_ids
    if missing_from_log:
        console.print(f"[bold red]Found {len(missing_from_log)} documents in images but not in mail log:[/]")
        for doc_id in sorted(missing_from_log):
            console.print(f"  - {doc_id}")
    
    # Check for documents in log but not in images
    missing_images = mail_log_doc_ids - image_doc_ids
    if missing_images:
        console.print(f"[bold yellow]Found {len(missing_images)} documents in mail log but not in images:[/]")
        for doc_id in sorted(missing_images):
            console.print(f"  - {doc_id}")
    
    # Documents to process (those in both mail log and images)
    docs_to_process = mail_log_doc_ids.intersection(image_doc_ids)
    console.print(f"[green]Will re-scan {len(docs_to_process)} documents[/]")
    
    # Ask for confirmation
    if not input("Proceed with re-scanning using Gemini? (y/n): ").lower().startswith('y'):
        console.print("[yellow]Aborted.[/]")
        return
    
    # Process each document
    results = []
    docs_with_discrepancies = 0
    
    for doc_id in sorted(docs_to_process):
        # Get mail log data for this document
        mail_log_data = mail_log_dict[doc_id]
        
        # Re-scan document with Gemini
        rescan_data = load_and_process_document(doc_id, doc_images[doc_id])
        
        if rescan_data:
            # Compare data
            comparison = compare_document_data(doc_id, rescan_data, mail_log_data)
            
            # Check if there are any mismatches
            has_mismatch = any(not data['match'] for data in comparison.values())
            
            # Display results
            display_comparison_results(comparison, doc_id)
            
            # Track results
            results.append({
                'doc_id': doc_id,
                'comparison': comparison,
                'has_mismatch': has_mismatch
            })
            
            if has_mismatch:
                docs_with_discrepancies += 1
    
    # Display summary
    console.print("\n[bold]Re-scanning Summary:[/]")
    console.print(f"Total documents re-scanned: {len(results)}")
    console.print(f"Documents with discrepancies: {docs_with_discrepancies}")
    
    # Show overall verification result
    if docs_with_discrepancies == 0 and not missing_from_log and not missing_images:
        console.print("[bold green]✓ Mail log is 100% accurate! All images are properly analyzed and recorded.[/]")
    else:
        console.print("[bold yellow]⚠ Discrepancies found. Review the details above.[/]")
    
    console.print("[bold]Verification complete![/]")

if __name__ == "__main__":
    main()
