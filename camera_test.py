#!/usr/bin/env python3
"""
Camera Test Utility for Desk-Mail-Bot

This script tests if your webcam can be accessed by OpenCV.
If successful, it will display your webcam feed in a window.
Press ESC to exit.

Features:
- Tests camera access and permissions
- Lists all available cameras including iPhone Continuity Camera
- Allows selection of specific camera
- Provides platform-specific troubleshooting help

Use this script to troubleshoot camera permission issues before
running the main Desk-Mail-Bot application.
"""

import cv2
import sys
import platform
import os
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()

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

def test_camera(camera_index=0):
    """Test if camera can be accessed and display feed"""
    console.print("[bold]Camera Test Utility[/]")
    
    # On macOS, check for Continuity Camera
    system = platform.system()
    if system == "Darwin" and os.path.exists("Info.plist"):
        console.print("[bold green]Info.plist detected![/] iPhone Continuity Camera should work properly.")
    
    # List available cameras
    cameras = list_available_cameras()
    
    if not cameras:
        console.print("[bold red]No cameras detected![/]")
        print_camera_permissions_help()
        return False
    
    console.print(f"[bold green]{len(cameras)} camera(s) found:[/]")
    for idx, (cam_id, name) in enumerate(cameras):
        console.print(f"  {idx+1}. {name} (device #{cam_id})")
    
    # If multiple cameras and interactive mode, ask user to select
    if len(cameras) > 1:
        try:
            selection = Prompt.ask(
                "Select camera to test",
                choices=[str(i+1) for i in range(len(cameras))],
                default="1"
            )
            camera_index = cameras[int(selection)-1][0]
        except (KeyboardInterrupt, EOFError):
            camera_index = cameras[0][0]
            console.print(f"Using default camera #{camera_index}")
    else:
        camera_index = cameras[0][0]
    
    console.print(f"Attempting to access camera #{camera_index}...")
    
    # Try to open the camera
    cam = cv2.VideoCapture(camera_index)
    
    if not cam.isOpened():
        console.print("[bold red]ERROR: Could not access camera![/]")
        print_camera_permissions_help()
        return False
    
    # Camera opened successfully
    console.print("[bold green]SUCCESS: Camera accessed![/]")
    console.print("Displaying camera feed in a new window.")
    console.print("Press [bold]ESC[/] to exit.")
    
    # Display camera feed
    while True:
        ret, frame = cam.read()
        
        if not ret:
            console.print("[bold red]ERROR: Failed to grab frame![/]")
            break
        
        cv2.imshow("Camera Test (ESC to exit)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    # Release resources
    cam.release()
    cv2.destroyAllWindows()
    console.print("[bold green]Camera test completed.[/]")
    return True

def print_camera_permissions_help():
    """Print OS-specific help for camera permissions"""
    system = platform.system()
    
    console.print("\n[bold yellow]Camera Permission Help:[/]")
    
    if system == "Darwin":  # macOS
        console.print("""
[bold]macOS Camera Permission Steps:[/]

1. Open [bold]System Settings[/] (or System Preferences)
2. Go to [bold]Privacy & Security[/] → [bold]Camera[/]
3. Make sure [bold]Terminal[/] or your code editor (e.g., VS Code) is allowed
4. If you just allowed it, [bold]restart Terminal[/] or your code editor
5. Run this test script again

Note: On newer macOS versions, you'll get a popup asking for camera
permission the first time. If you denied it, you'll need to enable
it manually in System Settings.
""")
    elif system == "Windows":
        console.print("""
[bold]Windows Camera Permission Steps:[/]

1. Open [bold]Settings[/]
2. Go to [bold]Privacy[/] → [bold]Camera[/]
3. Ensure [bold]"Allow apps to access your camera"[/] is ON
4. Make sure your terminal or code editor is allowed
5. Run this test script again
""")
    elif system == "Linux":
        console.print("""
[bold]Linux Camera Permission Help:[/]

Camera permissions on Linux vary by distribution, but try:

1. Ensure your user account is in the 'video' group:
   [bold]sudo usermod -a -G video $USER[/]
2. Log out and log back in for group changes to take effect
3. Check if other applications can access your camera
4. Try running the script with elevated permissions (not recommended for regular use):
   [bold]sudo python camera_test.py[/]
""")
    else:
        console.print("Please check camera permissions for your operating system.")

if __name__ == "__main__":
    test_camera()
