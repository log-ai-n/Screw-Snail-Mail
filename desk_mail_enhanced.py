#!/usr/bin/env python3
"""
Show‑and‑Tell Mail – Enhanced Edition
1. Opens your default camera
2. Press SPACE to capture the current frame
3. Auto-detects and crops the paper region
4. OCR → Gemini summary → CSV log + on‑screen print
5. Voice confirmation of processed mail
6. Support for multi-page documents

Controls:
- SPACE: Capture current frame
- P: Add additional page to current document
- N: Finish current document and start a new one
- ESC: Quit

Enhancements:
- Auto-crop for better OCR
- Voice feedback
- Multiple page support
- Improved UI with rich
"""

import cv2
import csv
import yaml
import os
import sys
import platform
import textwrap
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pytesseract
import google.generativeai as genai
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from datetime import datetime

# Load environment variables and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Constants
LOG = Path("mail_log.csv")
console = Console()

class MailDocument:
    """Class to handle multi-page documents"""
    
    def __init__(self):
        self.pages = []
        self.text = ""
        self.summary = None
    
    def add_page(self, text):
        """Add a page to the document"""
        self.pages.append(text)
        self.text = "\n\n----- PAGE BREAK -----\n\n".join(self.pages)
    
    def get_summary(self):
        """Get a summary from Gemini"""
        if not self.text:
            return None
        
        self.summary = summarise(self.text)
        return self.summary

def summarise(text: str) -> dict:
    """Extract key information from the letter text using Gemini"""
    prompt = f"""
    Summarise this physical letter. Extract:
      Sender, DocumentDate, Category (bill|statement|personal|ad|legal|other),
      AmountDue (if any), ActionNeeded (pay|reply|file|read|none).
    Give YAML only.

    ---\n{text[:8000]}\n---
    """
    r = genai.GenerativeModel("gemini-1.0-pro-latest").generate_content(prompt)
    try:
        data = yaml.safe_load(r.text)
        # Add capture timestamp
        data['ProcessedDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return data
    except Exception as e:
        console.print(f"[bold red]Error parsing Gemini response:[/] {e}")
        console.print(f"Raw response: {r.text}")
        return {
            "Sender": "ERROR - See log",
            "DocumentDate": "Unknown",
            "Category": "other",
            "AmountDue": "Unknown",
            "ActionNeeded": "review",
            "ProcessedDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def append(row: dict):
    """Append a row to the CSV log file"""
    first = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if first: w.writeheader()
        w.writerow(row)

def ocr(image):
    """Extract text from an image using Tesseract OCR"""
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing image...", total=3)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        progress.update(task, advance=1)
        
        # Apply threshold to improve OCR accuracy
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        progress.update(task, advance=1)
        
        # Perform OCR
        text = pytesseract.image_to_string(gray)
        progress.update(task, advance=1)
    
    return text

def find_document_corners(image):
    """Find and return the corners of a document in the image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    
    if largest_contour is None:
        return None
    
    # Approximate the contour to get a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we have a quadrilateral, return its corners
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    # If no suitable quadrilateral found, return None
    return None

def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Compute the difference between the points
    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """Apply a perspective transform to obtain a top-down view of the document"""
    # Obtain a consistent order of the points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct the set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def speak(text):
    """Use macOS say command to speak text"""
    os.system(f'say "{text}"')

def display_help(console):
    """Display help information"""
    table = Table(title="Show-and-Tell Mail Controls")
    table.add_column("Key", style="cyan")
    table.add_column("Action", style="green")
    
    table.add_row("SPACE", "Capture current frame")
    table.add_row("P", "Add additional page to current document")
    table.add_row("N", "Finish current document and start a new one")
    table.add_row("H", "Show this help")
    table.add_row("ESC", "Quit application")
    
    console.print(table)

def display_summary(data, console):
    """Display a rich formatted summary of the mail data"""
    if not data:
        console.print("[bold red]No data available to display[/]")
        return
    
    table = Table(title=f"Mail from {data.get('Sender', 'Unknown')}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in data.items():
        if key != 'Sender':  # Already in the title
            table.add_row(key, str(value))
    
    console.print(table)

def list_available_cameras():
    """List all available camera devices"""
    console.print("[bold]Scanning for camera devices...[/]")
    
    available_cameras = []
    max_cameras_to_check = 10
    
    # On macOS, check if we can try to directly get continuity camera
    system = platform.system()
    
    # Suppress OpenCV warnings during camera scanning
    # This prevents the "device out of bound" warnings from cluttering the output
    original_stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as null_stream:
            sys.stderr = null_stream
            
            for i in range(max_cameras_to_check):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # Get camera name if possible
                            name = f"Camera #{i}"
                            if hasattr(cap, 'getBackendName'):
                                backend = cap.getBackendName()
                                name = f"Camera #{i} ({backend})"
                                
                                # On macOS, add hint for Continuity Camera
                                if system == "Darwin" and "AVFOUNDATION" in backend and i > 0:
                                    name += " (Possibly iPhone)"
                            
                            available_cameras.append((i, name))
                    cap.release()
                except Exception as e:
                    # Just skip this camera index if there's an error
                    pass
    finally:
        # Restore stderr
        sys.stderr = original_stderr
    
    return available_cameras

def main():
    """Main application function"""
    # Check for Continuity Camera support
    if os.path.exists("Info.plist"):
        console.print("[bold green]Info.plist detected![/] iPhone Continuity Camera should be supported.")
    else:
        console.print("[yellow]Info.plist not found. iPhone Continuity Camera may not work properly.[/]")
        console.print("[yellow]If using an iPhone as webcam, run from the directory with Info.plist.[/]")
    
    # List available cameras
    cameras = list_available_cameras()
    
    if not cameras:
        console.print("[bold red]Error: No cameras detected![/]")
        console.print("[yellow]Tip: Check camera permissions in System Settings → Privacy → Camera[/]")
        console.print("[yellow]Make sure to allow Terminal or your code editor to access the camera.[/]")
        console.print("[yellow]If using macOS, you may need to restart Terminal or your editor after granting permission.[/]")
        return
    
    console.print(f"[bold green]{len(cameras)} camera(s) found:[/]")
    for idx, (cam_id, name) in enumerate(cameras):
        console.print(f"  {idx+1}. {name}")
    
    # Select camera
    camera_index = cameras[0][0]  # Default to first camera
    if len(cameras) > 1:
        try:
            from rich.prompt import Prompt
            selection = Prompt.ask(
                "Select camera to use",
                choices=[str(i+1) for i in range(len(cameras))],
                default="1"
            )
            camera_index = cameras[int(selection)-1][0]
        except (ImportError, KeyboardInterrupt, EOFError):
            console.print(f"Using default camera #{camera_index}")
    
    # Initialize camera
    console.print(f"[bold]Initializing camera #{camera_index}...[/]")
    cam = cv2.VideoCapture(camera_index)
    
    if not cam.isOpened():
        console.print("[bold red]Error: Could not open camera.[/]")
        console.print("[yellow]Tip: Check camera permissions in System Settings → Privacy → Camera[/]")
        console.print("[yellow]Make sure to allow Terminal or your code editor to access the camera.[/]")
        console.print("[yellow]If using macOS, you may need to restart Terminal or your editor after granting permission.[/]")
        return
    
    console.print(Panel("[bold green]Camera ready.[/]\nPoint a letter at the camera and press SPACE to capture.",
                       title="Show-and-Tell Mail", subtitle="ESC to quit, H for help"))
    
    current_document = MailDocument()
    capturing_multipage = False
    
    while True:
        ret, frame = cam.read()
        
        if not ret:
            console.print("[bold red]Error: Failed to grab frame from camera.[/]")
            break
        
        # Draw a rectangle to guide document placement
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (width//6, height//6), (width*5//6, height*5//6), (0, 255, 0), 2)
        
        # Show status text
        status_text = "CAPTURING MULTI-PAGE DOCUMENT" if capturing_multipage else "READY TO CAPTURE"
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=capture  P=add page  N=new doc  H=help  ESC=quit", 
                   (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("MailCam", frame)
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:  # ESC
            break
        
        elif k == 32:  # SPACE
            console.print("[yellow]⏳ Capturing…[/]")
            
            # Try to find and crop document
            corners = find_document_corners(frame)
            
            if corners is not None and len(corners) == 4:
                # We found a document, crop and transform
                warped = four_point_transform(frame, corners.astype(np.float32))
                # Show the cropped document
                cv2.imshow("Cropped Document", warped)
                text = ocr(warped)
            else:
                # No document detected, use full frame
                console.print("[yellow]No document detected, using full frame.[/]")
                text = ocr(frame)
            
            if not capturing_multipage:
                # Start a new document
                current_document = MailDocument()
                capturing_multipage = True
                
            # Add the page to the current document
            current_document.add_page(text)
            console.print(f"[green]✓ Page {len(current_document.pages)} captured[/]")
            speak(f"Page {len(current_document.pages)} captured")
        
        elif k == ord('p') or k == ord('P'):  # P key for multiple pages
            if capturing_multipage:
                console.print("[yellow]Ready for next page. Press SPACE to capture.[/]")
            else:
                console.print("[yellow]No active document. Press SPACE to start a new document.[/]")
        
        elif k == ord('n') or k == ord('N'):  # N key to finish the document
            if capturing_multipage and current_document.pages:
                # Process the complete document
                console.print("[yellow]⏳ Processing complete document…[/]")
                data = current_document.get_summary()
                
                if data:
                    append(data)
                    display_summary(data, console)
                    
                    # Speak a summary
                    action_needed = data.get('ActionNeeded', 'unknown action')
                    amount_due = data.get('AmountDue', 'no amount')
                    if amount_due not in ['', 'N/A', None, 'no amount']:
                        speak(f"Letter from {data.get('Sender', 'unknown')} needs {action_needed} with amount {amount_due}")
                    else:
                        speak(f"Letter from {data.get('Sender', 'unknown')} needs {action_needed}")
                
                # Reset for next document
                capturing_multipage = False
                console.print("[green]✓ Document processed and logged[/]")
            else:
                console.print("[yellow]No active document to process.[/]")
        
        elif k == ord('h') or k == ord('H'):  # H key for help
            display_help(console)
    
    # Clean up
    cam.release()
    cv2.destroyAllWindows()
    console.print("[bold green]Session complete.[/] Mail log saved to mail_log.csv")

if __name__ == "__main__":
    main()
