#!/usr/bin/env python3
"""
Mail Log Cleanup Tool

This script removes duplicate entries from the mail log file, keeping only the most
recent entry for each document ID, effectively cleaning up any error entries that
have since been correctly processed.
"""

import pandas as pd
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Constants
LOG_DIR = Path("mail_data")
LOG_FILE = LOG_DIR / "mail_log.csv"
BACKUP_FILE = LOG_DIR / "mail_log_backup.csv"
console = Console()

def backup_log_file():
    """Create a backup of the log file before modifying it"""
    if LOG_FILE.exists():
        import shutil
        shutil.copy2(LOG_FILE, BACKUP_FILE)
        console.print(f"[green]Created backup at {BACKUP_FILE}[/]")
        return True
    else:
        console.print(f"[bold red]Mail log file not found: {LOG_FILE}[/]")
        return False

def clean_log_file():
    """Clean up the mail log file by removing duplicates and error entries"""
    # Read the CSV file - try different delimiters
    try:
        # First try normal CSV
        df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
        original_rows = len(df)
        console.print(f"[green]Loaded mail log with {original_rows} entries (comma delimiter)[/]")
    except Exception as e:
        console.print(f"[yellow]Error with comma delimiter: {str(e)}[/]")
        try:
            # Try tab delimiter
            df = pd.read_csv(LOG_FILE, sep='\t', on_bad_lines='skip')
            original_rows = len(df)
            console.print(f"[green]Loaded mail log with {original_rows} entries (tab delimiter)[/]")
        except Exception as e:
            console.print(f"[yellow]Error with tab delimiter: {str(e)}[/]")
            try:
                # Try reading the file directly and fixing the format
                with open(LOG_FILE, 'r') as f:
                    lines = f.readlines()
                
                # Print the first few lines to debug
                console.print("[yellow]First 3 lines of the file:[/]")
                for i, line in enumerate(lines[:3]):
                    console.print(f"Line {i+1}: {line.strip()}")
                
                # Try to insert commas manually if needed
                fixed_lines = []
                for line in lines:
                    # Add commas between fields based on expected position
                    modified = line.replace('Sender', 'Sender,') \
                                 .replace('DocumentDate', 'DocumentDate,') \
                                 .replace('Category', 'Category,') \
                                 .replace('AmountDue', 'AmountDue,') \
                                 .replace('DueDate', 'DueDate,') \
                                 .replace('ActionNeeded', 'ActionNeeded,') \
                                 .replace('Overview', 'Overview,') \
                                 .replace('ProcessedDate', 'ProcessedDate,') \
                                 .replace('PageCount', 'PageCount,') \
                                 .replace('ImagePaths', 'ImagePaths,') \
                                 .replace('DocumentID', 'DocumentID')
                    fixed_lines.append(modified)
                
                # Write to a temporary file
                temp_file = LOG_DIR / "temp_mail_log.csv"
                with open(temp_file, 'w') as f:
                    f.writelines(fixed_lines)
                
                # Try to read the fixed file
                df = pd.read_csv(temp_file, on_bad_lines='skip')
                original_rows = len(df)
                console.print(f"[green]Loaded mail log with {original_rows} entries (after manual fixing)[/]")
            except Exception as e:
                console.print(f"[bold red]All attempts to read the mail log failed: {str(e)}[/]")
                return False
    
    # Identify which entries have document IDs
    df_with_id = df[df['DocumentID'].notna()].copy()
    
    # Count unique document IDs
    unique_ids = df_with_id['DocumentID'].nunique()
    console.print(f"[yellow]Found {unique_ids} unique document IDs[/]")
    
    # Group by DocumentID and keep only the last entry for each
    # This assumes more recent entries are better (e.g., fixed entries vs error entries)
    if not df_with_id.empty:
        df_clean = df_with_id.drop_duplicates(subset=['DocumentID'], keep='last')
        console.print(f"[green]Keeping {len(df_clean)} entries after removing duplicates[/]")
        
        # Find entries without DocumentIDs (if any are worth keeping)
        df_no_id = df[df['DocumentID'].isna()]
        
        # Only keep non-error entries without DocumentIDs
        df_no_id_clean = df_no_id[~df_no_id['Sender'].str.contains('ERROR', na=False)]
        console.print(f"[yellow]Found {len(df_no_id_clean)} entries without DocumentIDs worth keeping[/]")
        
        # Combine the clean sets
        df_final = pd.concat([df_clean, df_no_id_clean])
        
        # Sort by DocumentDate if available, otherwise by ProcessedDate
        if 'ProcessedDate' in df_final.columns:
            df_final = df_final.sort_values(by='ProcessedDate')
        elif 'DocumentDate' in df_final.columns:
            df_final = df_final.sort_values(by='DocumentDate')
        
        # Save the cleaned file
        df_final.to_csv(LOG_FILE, index=False)
        console.print(f"[bold green]✓ Cleaned mail log file saved with {len(df_final)} entries[/]")
        console.print(f"[green]Removed {original_rows - len(df_final)} duplicate/error entries[/]")
        
        # Display a table of the cleaned log file (first few entries)
        table = Table(title="Cleaned Mail Log (First 5 Entries)")
        
        # Add columns
        for col in df_final.columns[:6]:  # Limit to first 6 columns for display
            table.add_column(col, style="cyan")
        
        # Add rows (first 5)
        for _, row in df_final.head(5).iterrows():
            table.add_row(*[str(row[col]) for col in df_final.columns[:6]])
        
        console.print(table)
        
        return True
    else:
        console.print("[yellow]No entries with DocumentIDs found. Nothing to clean.[/]")
        return False

def main():
    """Main function"""
    console.print(Panel.fit(
        "[bold green]Mail Log Cleanup Tool[/]\n"
        "This tool removes duplicate entries from the mail log",
        title="Desk-Mail-Bot"
    ))
    
    # Create backup first
    if not backup_log_file():
        return
    
    # Clean the log file
    if clean_log_file():
        console.print("[bold green]Mail log cleanup complete![/]")
    else:
        console.print("[bold red]Mail log cleanup failed![/]")
        console.print(f"[yellow]Original file has been backed up to: {BACKUP_FILE}[/]")

if __name__ == "__main__":
    main()
