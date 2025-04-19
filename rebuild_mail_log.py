#!/usr/bin/env python3
"""
Mail Log Rebuild Tool

This script creates a fresh mail log by scanning the mail_data/images directory
and creating entries for each document based on the latest processed data.
It completely ignores the current mail_log.csv and builds a clean one from scratch.

This enhanced version also uses the desk_mail_vision module to reanalyze
each document using the Gemini Vision API.
"""

import os
import csv
import re
import cv2
from pathlib import Path
import glob
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from collections import defaultdict
from datetime import datetime

# Import the necessary functions from the desk_mail_vision module
from desk_mail_vision import analyze_multipage_mail, get_default_data_multipage

# Constants
LOG_DIR = Path("mail_data")
LOG_FILE = LOG_DIR / "mail_log.csv"
IMAGES_DIR = LOG_DIR / "images"
NEW_LOG_FILE = LOG_DIR / "mail_log_new.csv"
ERROR_LOG = LOG_DIR / "error_log.json"
console = Console()

def extract_doc_id(filename):
    """Extract document ID from filename"""
    # Extract the base document ID from filenames like "20250419_090014_ERR_page1.jpg"
    match = re.match(r'(\d{8}_\d{6}_[A-Za-z_]+)(?:_page\d+)?\.jpg', filename)
    if match:
        return match.group(1)
    
    # Handle simpler format like "20250419_093643__page1.jpg"
    match = re.match(r'(\d{8}_\d{6})(?:__page\d+)?\.jpg', filename)
    if match:
        return match.group(1)
    
    return None

def group_images_by_document():
    """Scan the images directory and group images by document ID"""
    if not IMAGES_DIR.exists():
        console.print(f"[bold red]Images directory not found: {IMAGES_DIR}[/]")
        return {}
    
    doc_images = defaultdict(list)
    
    # Walk through all files in the images directory
    for filepath in glob.glob(str(IMAGES_DIR / "*.jpg")):
        filename = os.path.basename(filepath)
        doc_id = extract_doc_id(filename)
        
        if doc_id:
            doc_images[doc_id].append(filepath)
    
    # Sort image paths for each document
    for doc_id in doc_images:
        doc_images[doc_id] = sorted(doc_images[doc_id])
    
    console.print(f"[green]Found {len(doc_images)} unique documents across {sum(len(imgs) for imgs in doc_images.values())} images[/]")
    
    # Display the first few documents and their image counts
    table = Table(title="Document Summary")
    table.add_column("Document ID", style="cyan")
    table.add_column("Image Count", style="green")
    
    for doc_id, image_paths in list(doc_images.items())[:5]:
        table.add_row(doc_id, str(len(image_paths)))
    
    console.print(table)
    
    return doc_images

def build_document_data(doc_id, image_paths):
    """Build document data for a given document ID"""
    # Basic document data structure with defaults
    data = {
        "Sender": doc_id.split('_')[-1] if '_' in doc_id else "Unknown",
        "DocumentDate": "Unknown",
        "Category": "Unknown",
        "AmountDue": "Unknown",
        "DueDate": "Unknown",
        "ActionNeeded": "Unknown",
        "Overview": f"Document with {len(image_paths)} pages",
        "ProcessedDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "PageCount": len(image_paths),
        "ImagePaths": ";".join(image_paths),
        "DocumentID": doc_id
    }
    
    # Try to get the processed date from filename
    if doc_id.startswith("20"):
        year = doc_id[0:4]
        month = doc_id[4:6]
        day = doc_id[6:8]
        data["ProcessedDate"] = f"{year}-{month}-{day} {doc_id[9:11]}:{doc_id[11:13]}:{doc_id[13:15]}"
    
    # Load images and analyze with Gemini Vision API
    try:
        console.print(f"[yellow]Analyzing document {doc_id} with Gemini Vision API...[/]")
        
        # Load images using OpenCV
        images = []
        for path in sorted(image_paths):
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
        
        if images:
            # Process the document using the desk_mail_vision module
            vision_data = analyze_multipage_mail(images)
            
            if vision_data:
                # Update our data with the vision results
                for key, value in vision_data.items():
                    if key != "ImagePaths" and key != "DocumentID":  # Keep our original values for these
                        data[key] = value
                
                console.print(f"[green]Successfully analyzed document {doc_id} with Gemini Vision API[/]")
            else:
                console.print(f"[yellow]Could not extract data from document {doc_id} using Gemini Vision API[/]")
        else:
            console.print(f"[yellow]No valid images found for document {doc_id}[/]")
    except Exception as e:
        console.print(f"[red]Error analyzing document {doc_id}: {str(e)}[/]")
    
    # As a fallback, get info from error log if available and if we don't have good data
    if data["Sender"] == "Unknown" and ERROR_LOG.exists():
        try:
            with open(ERROR_LOG, 'r') as f:
                errors = json.load(f)
                
            # Look for entries for this document
            for error in errors:
                if "doc_id" in error and error["doc_id"] == doc_id:
                    # Extract structured data from raw text if available
                    if "raw_text" in error:
                        raw_text = error["raw_text"]
                        
                        # Try to extract sender
                        sender_match = re.search(r"Sender: ([^\n]+)", raw_text)
                        if sender_match and data["Sender"] == "Unknown":
                            data["Sender"] = sender_match.group(1).strip()
                        
                        # Try to extract amount
                        amount_match = re.search(r"AmountDue: ([^\n]+)", raw_text)
                        if amount_match and data["AmountDue"] == "Unknown":
                            data["AmountDue"] = amount_match.group(1).strip()
                        
                        # Try to extract due date
                        due_date_match = re.search(r"DueDate: ([^\n]+)", raw_text)
                        if due_date_match and data["DueDate"] == "Unknown":
                            data["DueDate"] = due_date_match.group(1).strip()
                        
                        # Try to extract category
                        category_match = re.search(r"Category: ([^\n]+)", raw_text)
                        if category_match and data["Category"] == "Unknown":
                            data["Category"] = category_match.group(1).strip()
                        
                        # Try to extract action needed
                        action_match = re.search(r"ActionNeeded: ([^\n]+)", raw_text)
                        if action_match and data["ActionNeeded"] == "Unknown":
                            data["ActionNeeded"] = action_match.group(1).strip()
                        
                        # Try to extract overview
                        overview_match = re.search(r"Overview: ([^\n]+)", raw_text)
                        if overview_match and (data["Overview"] == f"Document with {len(image_paths)} pages" or data["Overview"] == "Unknown"):
                            data["Overview"] = overview_match.group(1).strip()
        except Exception as e:
            console.print(f"[yellow]Error reading error log: {str(e)}[/]")
    
    return data

def create_new_log_file(document_data):
    """Create a new mail log file with the compiled document data"""
    if not document_data:
        console.print("[bold red]No document data available to create log file[/]")
        return False
    
    # Get all fields from all documents
    fields = set()
    for data in document_data.values():
        fields.update(data.keys())
    
    # Ensure essential fields are present and in the right order
    essential_fields = ["Sender", "DocumentDate", "Category", "AmountDue", "DueDate", 
                       "ActionNeeded", "Overview", "ProcessedDate", "PageCount", 
                       "ImagePaths", "DocumentID"]
    
    # Create a sorted field list with essential fields first, followed by any other fields
    field_order = [f for f in essential_fields if f in fields]
    field_order += sorted([f for f in fields if f not in essential_fields])
    
    try:
        with NEW_LOG_FILE.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_order)
            writer.writeheader()
            
            # Write each document's data
            for doc_id, data in document_data.items():
                writer.writerow({field: data.get(field, "") for field in field_order})
        
        console.print(f"[bold green]✓ Created new mail log with {len(document_data)} entries at {NEW_LOG_FILE}[/]")
        
        # Display the first few entries in the new log
        table = Table(title="New Mail Log (First 5 Entries)")
        
        # Add columns (limit to first 6 for readability)
        for field in field_order[:6]:
            table.add_column(field, style="cyan")
        
        # Add rows (first 5)
        for i, (doc_id, data) in enumerate(document_data.items()):
            if i >= 5:
                break
            table.add_row(*[str(data.get(field, "")) for field in field_order[:6]])
        
        console.print(table)
        
        return True
    except Exception as e:
        console.print(f"[bold red]Error creating new mail log: {str(e)}[/]")
        return False

def replace_old_log_with_new():
    """Replace the old mail log with the new one"""
    if not NEW_LOG_FILE.exists():
        console.print("[bold red]New mail log file not found[/]")
        return False
    
    try:
        # Create a backup of the old log file if it exists
        if LOG_FILE.exists():
            backup_file = LOG_DIR / f"mail_log_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            import shutil
            shutil.copy2(LOG_FILE, backup_file)
            console.print(f"[green]Created backup of old log at {backup_file}[/]")
        
        # Replace the old log with the new one
        if NEW_LOG_FILE.exists():
            os.replace(NEW_LOG_FILE, LOG_FILE)
            console.print(f"[bold green]✓ Replaced old mail log with new one[/]")
            return True
        else:
            console.print(f"[bold red]New mail log file not found: {NEW_LOG_FILE}[/]")
            return False
    except Exception as e:
        console.print(f"[bold red]Error replacing mail log: {str(e)}[/]")
        return False

def main():
    """Main function"""
    console.print(Panel.fit(
        "[bold green]Mail Log Rebuild Tool[/]\n"
        "This tool creates a fresh mail log by scanning the images directory",
        title="Desk-Mail-Bot"
    ))
    
    # Group images by document
    doc_images = group_images_by_document()
    if not doc_images:
        return
    
    # Build document data for each document
    console.print("[yellow]Building document data from images...[/]")
    document_data = {}
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing documents...", total=len(doc_images))
        
        for doc_id, image_paths in doc_images.items():
            document_data[doc_id] = build_document_data(doc_id, image_paths)
            progress.update(task, advance=1)
    
    # Create a new mail log file
    if create_new_log_file(document_data):
        # Replace the old log with the new one
        if replace_old_log_with_new():
            console.print("[bold green]Mail log rebuild complete![/]")
        else:
            console.print("[bold red]Failed to replace old mail log with new one[/]")
    else:
        console.print("[bold red]Failed to create new mail log[/]")

if __name__ == "__main__":
    main()
