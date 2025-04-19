#!/usr/bin/env python3
"""
Complete Image Verification Tool for Desk-Mail-Bot

This script ensures that every single image file in mail_data/images 
is properly referenced in the mail_log.csv file.
"""

import os
import csv
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import re

# Constants
IMAGES_DIR = Path("mail_data/images")
LOG_FILE = Path("mail_data/mail_log.csv")
console = Console()

def validate_page_numbers(doc_id, image_files):
    """Check if page numbers in a document are sequential without gaps"""
    # Filter to just files matching this doc_id
    doc_files = [f for f in image_files if doc_id in f]
    
    # Extract page numbers
    page_numbers = []
    for file in doc_files:
        match = re.search(r'page(\d+)\.', file)
        if match:
            page_numbers.append(int(match.group(1)))
    
    if not page_numbers:
        return True, "No page numbers found"
    
    # Sort and check for gaps
    page_numbers.sort()
    expected_pages = list(range(1, max(page_numbers) + 1))
    missing_pages = set(expected_pages) - set(page_numbers)
    
    if missing_pages:
        return False, f"Missing pages: {sorted(missing_pages)}"
    return True, "All pages sequential"

def main():
    """Main function"""
    console.print(Panel.fit(
        "[bold green]Complete Image Verification Tool[/]\n"
        "This tool checks that every image file is referenced in the mail log",
        title="Desk-Mail-Bot"
    ))
    
    # Get all image files in the directory (excluding README.md)
    image_files = set()
    for filename in os.listdir(IMAGES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename != "README.md":
            image_files.add(str(Path("mail_data/images") / filename))
    
    # Print detailed list of files found
    console.print(f"[green]Found {len(image_files)} image files in the directory:[/]")
    file_table = Table(title="Image Files in Directory")
    file_table.add_column("Image Path", style="cyan")
    for img in sorted(image_files):
        file_table.add_row(img)
    console.print(file_table)
    
    # Read the mail log and extract all referenced image paths
    if not LOG_FILE.exists():
        console.print(f"[bold red]Mail log file not found: {LOG_FILE}[/]")
        return
    
    try:
        df = pd.read_csv(LOG_FILE)
        
        if 'ImagePaths' not in df.columns:
            console.print("[bold red]ImagePaths column not found in mail log[/]")
            return
        
        # Display document IDs in the mail log
        console.print(f"[green]Mail log contains {len(df)} documents:[/]")
        doc_table = Table(title="Documents in Mail Log")
        doc_table.add_column("Document ID", style="magenta")
        doc_table.add_column("# Pages", style="blue")
        doc_table.add_column("Image Paths", style="green")
        
        for idx, row in df.iterrows():
            doc_id = row.get('DocumentID', 'Unknown')
            paths = row.get('ImagePaths', '')
            num_pages = len(paths.split(';')) if isinstance(paths, str) else 0
            truncated_paths = (paths[:60] + '...') if isinstance(paths, str) and len(paths) > 60 else paths
            doc_table.add_row(str(doc_id), str(num_pages), str(truncated_paths))
        
        console.print(doc_table)
        
        # Extract all image paths from the mail log
        referenced_images = set()
        for paths in df['ImagePaths']:
            if isinstance(paths, str):
                for path in paths.split(';'):
                    path = path.strip()
                    if path:
                        referenced_images.add(path)
        
        console.print(f"[green]Found {len(referenced_images)} image references in mail log[/]")
        
        # Check for document ID consistency and sequence gaps
        console.print("[yellow]Checking document IDs and page sequences...[/]")
        doc_ids = set()
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # Try to extract document ID
            match = re.match(r'(\d{8}_\d{6}[^_]*)', filename)
            if match:
                doc_ids.add(match.group(1))
        
        sequence_table = Table(title="Document Page Sequence Check")
        sequence_table.add_column("Document ID", style="magenta")
        sequence_table.add_column("Status", style="green")
        sequence_table.add_column("Details", style="yellow")
        
        for doc_id in sorted(doc_ids):
            is_valid, message = validate_page_numbers(doc_id, image_files)
            status = "[green]OK[/]" if is_valid else "[red]Error[/]"
            sequence_table.add_row(doc_id, status, message)
        
        console.print(sequence_table)
        
        # Print specific image paths from mail log for comparison
        ref_table = Table(title="Sample of Referenced Images from Mail Log")
        ref_table.add_column("#", style="dim")
        ref_table.add_column("Image Path", style="green")
        
        sample_size = min(10, len(referenced_images))
        for i, img in enumerate(sorted(list(referenced_images)[:sample_size]), 1):
            ref_table.add_row(str(i), img)
        
        console.print(ref_table)
        
        # Check for images in directory but not in log
        missing_references = image_files - referenced_images
        
        # Check for images in log but not in directory
        missing_files = referenced_images - image_files
        
        # Display detailed results
        if not missing_references and not missing_files:
            console.print("[bold green]✓ All images are properly referenced in the mail log![/]")
            
            # Double check the count matches
            console.print(f"[green]Directory has {len(image_files)} images (excluding README.md)[/]")
            console.print(f"[green]Mail log references {len(referenced_images)} images[/]")
            
            # Verify each image is used only once
            image_count = {}
            for paths in df['ImagePaths']:
                if isinstance(paths, str):
                    for path in paths.split(';'):
                        path = path.strip()
                        if path:
                            image_count[path] = image_count.get(path, 0) + 1
            
            duplicates = {img: count for img, count in image_count.items() if count > 1}
            if duplicates:
                console.print("[bold yellow]Warning: Some images are referenced multiple times:[/]")
                dup_table = Table(title="Images Referenced Multiple Times")
                dup_table.add_column("Image Path", style="yellow")
                dup_table.add_column("Count", style="red")
                for img, count in sorted(duplicates.items()):
                    dup_table.add_row(img, str(count))
                console.print(dup_table)
        else:
            # Images missing from log
            if missing_references:
                console.print(f"[bold red]Found {len(missing_references)} images not referenced in mail log:[/]")
                table = Table(title="Images Not Referenced in Mail Log")
                table.add_column("Image Path", style="red")
                for img in sorted(missing_references):
                    table.add_row(img)
                console.print(table)
            
            # Referenced images missing from directory
            if missing_files:
                console.print(f"[bold yellow]Found {len(missing_files)} referenced images missing from directory:[/]")
                table = Table(title="Referenced Images Missing from Directory")
                table.add_column("Image Path", style="yellow")
                for img in sorted(missing_files):
                    table.add_row(img)
                console.print(table)
    
    except Exception as e:
        console.print(f"[bold red]Error processing mail log: {str(e)}[/]")

if __name__ == "__main__":
    main()
