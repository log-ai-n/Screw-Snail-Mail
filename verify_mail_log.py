#!/usr/bin/env python3
"""
Mail Log Verification Tool for Desk-Mail-Bot

This script checks all images in the mail_data/images folder and ensures they're
properly recorded in the mail_log.csv file. It can also process any missing 
documents using the Gemini Vision API.
"""

import os
import csv
import re
import cv2
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Import the necessary functions from the main script
from desk_mail_vision import (
    analyze_multipage_mail, 
    encode_image, 
    append,
    save_images,
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

def get_recorded_doc_ids(df):
    """Get set of document IDs already recorded in the mail log"""
    if df.empty or 'DocumentID' not in df.columns:
        return set()
    
    return set(df['DocumentID'].dropna().unique())

def scan_image_directory():
    """Scan the images directory and group images by document ID"""
    if not IMAGES_DIR.exists():
        console.print(f"[bold red]Images directory not found: {IMAGES_DIR}[/]")
        return {}
    
    doc_images = defaultdict(list)
    total_images = 0
    
    # Walk through all files in the images directory
    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        doc_id = extract_doc_id(filename)
        if doc_id:
            doc_images[doc_id].append(IMAGES_DIR / filename)
            total_images += 1
    
    console.print(f"[green]Found {total_images} images across {len(doc_images)} documents[/]")
    return doc_images

def find_unrecorded_docs(all_doc_ids, recorded_doc_ids):
    """Find document IDs that aren't recorded in the mail log"""
    return all_doc_ids - recorded_doc_ids

def load_and_process_document(doc_id, image_paths):
    """Load and process a document with its images"""
    console.print(f"[yellow]Processing document: {doc_id}[/]")
    
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
    
    # Process the document using the existing function
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
        
        # Create a table for the document details
        table = Table(title=f"Document {doc_id}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in data.items():
            if key != 'ImagePaths':  # Skip long image paths
                table.add_row(key, str(value))
        
        console.print(table)
        
        return data
    else:
        console.print(f"[bold red]Failed to process document: {doc_id}[/]")
        return None

def main():
    """Main function"""
    console.print(Panel.fit(
        "[bold green]Mail Log Verification Tool[/]\n"
        "This tool checks for any images not recorded in the mail log",
        title="Desk-Mail-Bot"
    ))
    
    # Load mail log
    mail_log = load_mail_log()
    recorded_doc_ids = get_recorded_doc_ids(mail_log)
    console.print(f"[green]Found {len(recorded_doc_ids)} document IDs in mail log[/]")
    
    # Scan image directory
    doc_images = scan_image_directory()
    all_doc_ids = set(doc_images.keys())
    
    # Find unrecorded documents
    unrecorded_doc_ids = find_unrecorded_docs(all_doc_ids, recorded_doc_ids)
    
    if not unrecorded_doc_ids:
        console.print("[bold green]✓ All documents in the images directory are recorded in the mail log![/]")
        return
    
    console.print(f"[yellow]Found {len(unrecorded_doc_ids)} documents not recorded in the mail log:[/]")
    for doc_id in unrecorded_doc_ids:
        console.print(f"  - {doc_id} ({len(doc_images[doc_id])} images)")
    
    # Ask if user wants to process unrecorded documents
    process_docs = input("Would you like to process these documents now? (y/n): ").lower().strip() == 'y'
    
    if process_docs:
        processed_count = 0
        
        for doc_id in unrecorded_doc_ids:
            data = load_and_process_document(doc_id, doc_images[doc_id])
            
            if data:
                # Append to mail log
                append(data)
                processed_count += 1
                console.print(f"[green]✓ Added {doc_id} to mail log[/]")
        
        console.print(f"[bold green]✓ Processed {processed_count} of {len(unrecorded_doc_ids)} unrecorded documents[/]")
    else:
        console.print("[yellow]Skipping document processing[/]")
    
    console.print("[bold]Verification complete![/]")

if __name__ == "__main__":
    main()
