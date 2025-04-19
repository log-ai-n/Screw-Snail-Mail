#!/usr/bin/env python3
"""
Show‑and‑Tell Mail – iPhone Continuity Camera Edition

This version is specifically optimized for using your iPhone as a camera
via Apple's Continuity Camera feature. It includes:

1. All features of the enhanced version
2. Special handling for iPhone camera detection and connection
3. Automatic reconnection if the camera connection is lost
4. Optimized settings for the high-quality iPhone camera

Requirements:
- macOS + iOS device with Continuity Camera enabled
- Info.plist file in the same directory as this script
"""

import cv2
import csv
import yaml
import os
import sys
import time
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
    table.add_row("R", "Reconnect to iPhone camera")
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

def find_iphone_camera():
    """Specifically look for iPhone Continuity Camera"""
    console.print("[bold]Looking for iPhone Continuity Camera...[/]")
    
    # Try all possible camera indices, iPhone is usually not the first one
    # but more likely to be index 1 or higher
    max_cameras_to_check = 10
    iphone_candidates = []
    
    # Suppress OpenCV warnings during camera scanning
    original_stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as null_stream:
            sys.stderr = null_stream
            
            # First check higher indices as iPhone is usually not camera 0
            for i in range(1, max_cameras_to_check):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # Successfully read a frame, this could be an iPhone
                            name = f"Camera #{i}"
                            is_likely_iphone = False
                            
                            # Try to get more info about the camera
                            if hasattr(cap, 'getBackendName'):
                                backend = cap.getBackendName()
                                name = f"Camera #{i} ({backend})"
                                
                                # On macOS, AVFOUNDATION backend with index > 0 is likely iPhone
                                if "AVFOUNDATION" in backend:
                                    name += " (Likely iPhone)"
                                    is_likely_iphone = True
                            
                            # Try to check resolution - iPhones have higher resolution
                            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            
                            if width > 1280 or height > 720:  # Higher than HD
                                is_likely_iphone = True
                                name += f" ({width}x{height})"
                            
                            # Add to candidates, with likely iPhones first
                            if is_likely_iphone:
                                iphone_candidates.insert(0, (i, name))
                            else:
                                iphone_candidates.append((i, name))
                    cap.release()
                except Exception:
                    pass
                    
            # Now check the default camera (0) as a fallback
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        name = "Default Camera #0"
                        if hasattr(cap, 'getBackendName'):
                            backend = cap.getBackendName()
                            name = f"Default Camera #0 ({backend})"
                        
                        # Add default camera last
                        iphone_candidates.append((0, name))
                cap.release()
            except Exception:
                pass
    finally:
        # Restore stderr
        sys.stderr = original_stderr
    
    return iphone_candidates

def connect_to_camera(camera_index):
    """Connect to a camera and verify it's working"""
    console.print(f"[bold]Connecting to camera #{camera_index}...[/]")
    
    # Initialize camera with higher resolution settings
    cam = cv2.VideoCapture(camera_index)
    
    # Try to set higher resolution for iPhone
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Verify connection
    if not cam.isOpened():
        console.print("[bold red]Failed to connect![/]")
        return None
    
    # Verify we can read frames
    ret, frame = cam.read()
    if not ret or frame is None:
        console.print("[bold red]Connected but can't read frames![/]")
        cam.release()
        return None
    
    # Success
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    console.print(f"[bold green]Successfully connected! Resolution: {width}x{height}[/]")
    
    return cam

def reconnect_camera(camera_index):
    """Try to reconnect to the camera if connection is lost"""
    for attempt in range(3):
        console.print(f"[yellow]Reconnection attempt {attempt+1}/3...[/]")
        cam = connect_to_camera(camera_index)
        if cam is not None:
            return cam
        time.sleep(1)  # Wait briefly between attempts
    
    console.print("[bold red]Failed to reconnect after multiple attempts.[/]")
    return None

def main():
    """Main application function"""
    # Check for Continuity Camera support via Info.plist
    if not os.path.exists("Info.plist"):
        console.print("[bold red]Error: Info.plist not found![/]")
        console.print("[yellow]iPhone Continuity Camera requires Info.plist in the same directory.[/]")
        console.print("[yellow]Please run this script from the directory containing Info.plist.[/]")
        return
    
    console.print(Panel(
        "[bold green]iPhone Continuity Camera Edition[/]\n"
        "This version is specially designed to work with your iPhone camera.",
        title="Show-and-Tell Mail"))
    
    # Look for iPhone camera
    console.print("[bold]Searching for iPhone camera...[/]")
    camera_options = find_iphone_camera()
    
    if not camera_options:
        console.print("[bold red]No cameras detected![/]")
        console.print("[yellow]Please ensure your iPhone is nearby and Continuity Camera is enabled.[/]")
        console.print("[yellow]iPhone settings: Settings → General → AirPlay & Handoff → Continuity Camera[/]")
        return
    
    # Choose a camera
    console.print(f"[bold green]{len(camera_options)} camera(s) found:[/]")
    for idx, (cam_id, name) in enumerate(camera_options):
        console.print(f"  {idx+1}. {name}")
    
    # If we found likely iPhone cameras or multiple cameras, let user choose
    camera_index = camera_options[0][0]  # Default to first option
    try:
        from rich.prompt import Prompt
        selection = Prompt.ask(
            "Select camera to use (1 is most likely to be iPhone)",
            choices=[str(i+1) for i in range(len(camera_options))],
            default="1"
        )
        camera_index = camera_options[int(selection)-1][0]
    except (ImportError, KeyboardInterrupt, EOFError):
        console.print(f"Using first camera option: {camera_options[0][1]}")
    
    # Connect to the selected camera
    cam = connect_to_camera(camera_index)
    if cam is None:
        console.print("[bold red]Failed to initialize camera. Please try again.[/]")
        return
    
    # Display startup message
    console.print(Panel(
        "[bold green]Camera ready.[/]\n"
        "Point a letter at the camera and press SPACE to capture.\n"
        "Press R to reconnect if camera connection is lost.",
        title="Show-and-Tell Mail",
        subtitle="ESC to quit, H for help"
    ))
    
    # Set up document state
    current_document = MailDocument()
    capturing_multipage = False
    
    # Main loop
    while True:
        # Read frame
        ret, frame = cam.read()
        
        # Handle connection loss
        if not ret or frame is None:
            console.print("[bold red]Camera connection lost![/]")
            console.print("[yellow]Press R to attempt reconnection or ESC to quit.[/]")
            
            # Wait for user input
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r') or key == ord('R'):
                    # Try to reconnect
                    cam.release()
                    cam = reconnect_camera(camera_index)
                    if cam is not None:
                        console.print("[bold green]Reconnected successfully![/]")
                        break
                    else:
                        console.print("[bold red]Reconnection failed. Press ESC to quit or R to try again.[/]")
            
            # If we couldn't reconnect and user pressed ESC, exit
            if not cam or not cam.isOpened():
                break
            continue
        
        # Draw a rectangle to guide document placement
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (width//6, height//6), (width*5//6, height*5//6), (0, 255, 0), 2)
        
        # Show status text
        status_text = "CAPTURING MULTI-PAGE DOCUMENT" if capturing_multipage else "READY TO CAPTURE"
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=capture  P=add page  N=new doc  H=help  R=reconnect  ESC=quit", 
                   (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow("MailCam (iPhone)", frame)
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
            
        elif k == ord('r') or k == ord('R'):  # R key to reconnect
            console.print("[yellow]Attempting to reconnect camera...[/]")
            cam.release()
            cam = reconnect_camera(camera_index)
            if cam is None:
                console.print("[bold red]Reconnection failed. Press ESC to quit or R to try again.[/]")
    
    # Clean up
    if cam is not None:
        cam.release()
    cv2.destroyAllWindows()
    console.print("[bold green]Session complete.[/] Mail log saved to mail_log.csv")

if __name__ == "__main__":
    main()
