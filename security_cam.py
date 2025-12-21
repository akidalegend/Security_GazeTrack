import cv2
import sys
import numpy as np
import datetime
import os
import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from PIL import Image, ImageTk
from ultralytics import YOLO

class SecuritySystem:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x600")
        
        # --- DESIGN: Dark Mode Theme Colors ---
        self.colors = {
            "bg": "#1e1e1e",       # Dark Grey Background
            "panel": "#252526",    # Slightly lighter panel
            "text": "#d4d4d4",     # Off-white text
            "accent": "#007acc",   # VS Code Blue
            "alert": "#d63031",    # Red for recording
            "success": "#00b894"   # Green for system ready
        }
        self.window.configure(bg=self.colors["bg"])

        # --- LOGIC SETUP ---
        print("Initializing AI Model...")
        self.model = YOLO('yolov8n.pt') 
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            sys.exit()

        self.is_recording = False
        self.out = None
        self.no_motion_frames = 0
        self.recording_cooldown = 30
        self.latest_frame = None # Initialize to None

        # --- GUI LAYOUT: Grid System ---
        # 1. Header Section
        self.header = tk.Frame(window, bg=self.colors["panel"], height=50)
        self.header.pack(fill=tk.X, side=tk.TOP)
        
        self.title_label = tk.Label(self.header, text="NORTHAMPTON WAREHOUSE SENTINEL", 
                                    font=("Helvetica", 16, "bold"), 
                                    bg=self.colors["panel"], fg=self.colors["accent"])
        self.title_label.pack(pady=10)

        # 2. Main Content Area (Holds Video + Sidebar)
        self.main_content = tk.Frame(window, bg=self.colors["bg"])
        self.main_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 3. Video Canvas (Left Side)
        # Added a border to make it look like a monitor
        self.video_frame = tk.Frame(self.main_content, bg="black", bd=2, relief="sunken")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.video_frame, bg="black", width=640, height=360)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 4. Sidebar Controls (Right Side)
        self.sidebar = tk.Frame(self.main_content, bg=self.colors["panel"], width=300)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        self.sidebar.pack_propagate(False) # Force width

        # Status Indicator
        self.status_lbl = tk.Label(self.sidebar, text="● SYSTEM ACTIVE", fg=self.colors["success"], bg=self.colors["panel"], font=("Arial", 10, "bold"))
        self.status_lbl.pack(pady=(20, 10))

        # --- TABS: Log & Recordings ---
        self.tabs = ttk.Notebook(self.sidebar)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tab 1: Log
        self.tab_log = tk.Frame(self.tabs, bg=self.colors["panel"])
        self.tabs.add(self.tab_log, text="System Log")
        
        self.log_text = tk.Text(self.tab_log, height=15, bg="black", fg="#00ff00", font=("Courier New", 10), bd=0)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Recordings
        self.tab_rec = tk.Frame(self.tabs, bg=self.colors["panel"])
        self.tabs.add(self.tab_rec, text="Recordings")

        self.rec_list = tk.Listbox(self.tab_rec, bg="black", fg="white", bd=0, selectbackground=self.colors["accent"])
        self.rec_list.pack(fill=tk.BOTH, expand=True, pady=5)
        
        btn_play = tk.Button(self.tab_rec, text="▶ PLAY SELECTED", command=self.play_recording, 
                             bg=self.colors["accent"], fg="white", font=("Arial", 9, "bold"))
        btn_play.pack(fill=tk.X)
        
        btn_refresh = tk.Button(self.tab_rec, text="⟳ REFRESH", command=self.refresh_recordings, 
                                bg="#444", fg="white", font=("Arial", 9))
        btn_refresh.pack(fill=tk.X)

        self.refresh_recordings() # Initial load

        # ROI Toggle
        self.use_roi = tk.BooleanVar(value=True)
        tk.Checkbutton(self.sidebar, text="Enable Restricted Zone (ROI)", variable=self.use_roi, 
                       bg=self.colors["panel"], fg="white", selectcolor="#444", activebackground=self.colors["panel"]).pack(pady=10)

        # Sensitivity Slider with Value Label
        tk.Label(self.sidebar, text="MOTION THRESHOLD", bg=self.colors["panel"], fg=self.colors["text"], font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(20, 5))
        
        # Value readout
        self.sens_value_lbl = tk.Label(self.sidebar, text="1000", bg=self.colors["panel"], fg=self.colors["accent"])
        self.sens_value_lbl.pack()

        self.sensitivity = ttk.Scale(self.sidebar, from_=500, to=5000, orient=tk.HORIZONTAL, command=self.update_sens_label)
        self.sensitivity.set(1000)
        self.sensitivity.pack(fill=tk.X, padx=10)

        # Buttons (Using ttk for Mac compatibility)
        style = ttk.Style()
        style.theme_use('clam') # Helps with styling on some OS
        
        self.btn_frame = tk.Frame(self.sidebar, bg=self.colors["panel"])
        self.btn_frame.pack(side=tk.BOTTOM, pady=20, fill=tk.X)
        
        self.btn_quit = tk.Button(self.btn_frame, text="SHUT DOWN SYSTEM", command=self.quit_app, 
                                  bg=self.colors["alert"], fg="white", font=("Arial", 11, "bold"),
                                  activebackground="#b71c1c", relief="flat", pady=10)
        self.btn_quit.pack(fill=tk.X, padx=10)

        self.update_loop()

    def update_sens_label(self, val):
        self.sens_value_lbl.config(text=f"{int(float(val))}")

    def log_message(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def start_recording(self, frame, frame_width, frame_height):
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
        filename = f"recordings/incident_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # MJPG is safer for Mac QuickTime than XVID
        self.out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))
        self.is_recording = True
        self.status_lbl.config(text="● RECORDING IN PROGRESS", fg=self.colors["alert"]) # Update Status
        self.log_message(f"REC STARTED")
        
        # Save the passed frame (which will be annotated)
        img_filename = filename.replace(".avi", ".jpg")
        cv2.imwrite(img_filename, frame) # Save the frame
        self.log_message(f"SNAPSHOT SAVED") # Save snapshot when recording starts for a mugshot (Smart addition)

    def stop_recording(self):
        if self.is_recording:
            self.out.release()
            self.is_recording = False
            self.status_lbl.config(text="● SYSTEM ACTIVE", fg=self.colors["success"]) # Reset Status
            self.log_message("REC STOPPED: Saved")

    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            self.latest_frame = frame # Store the frame for snapshots
            
            # Logic remains mostly same, just better integration
            mask = self.fgbg.apply(frame)
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

            height, width = mask.shape
            
            # --- FEATURE A: Define ROI (Region of Interest) ---
            # We define a box in the center/bottom where we CARE about motion
            roi_x1, roi_y1 = int(width * 0.1), int(height * 0.3) 
            roi_x2, roi_y2 = int(width * 0.9), int(height * 0.9)
            
            # Black out everything OUTSIDE the ROI if enabled
            if self.use_roi.get():
                roi_mask = np.zeros_like(mask)
                roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255 # White rectangle in ROI
                mask = cv2.bitwise_and(mask, roi_mask) # Only keep motion inside ROI

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            min_area = self.sensitivity.get()

            for cnt in contours:
                if cv2.contourArea(cnt) > min_area:
                    motion_detected = True
                    break 
            
            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()

            # Draw ROI on screen for UX
            if self.use_roi.get():
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "RESTRICTED ZONE", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if motion_detected:
                self.no_motion_frames = 0
                if not self.is_recording:
                    h, w, _ = frame.shape
                    self.start_recording(annotated_frame, w, h)
            else:
                self.no_motion_frames += 1
                if self.is_recording and self.no_motion_frames > self.recording_cooldown:
                    self.stop_recording()

            if self.is_recording:
                cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.out.write(annotated_frame)

            # Resize logic to fit the new canvas size dynamically
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            if canvas_w > 1 and canvas_h > 1: # Prevent error on startup
                annotated_frame = cv2.resize(annotated_frame, (canvas_w, canvas_h))

            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

        self.window.after(10, self.update_loop)

    def quit_app(self):
        if self.is_recording:
            self.stop_recording()
        self.cap.release()
        self.window.destroy()

    def refresh_recordings(self):
        self.rec_list.delete(0, tk.END)
        if os.path.exists("recordings"):
            files = sorted([f for f in os.listdir("recordings") if f.endswith(".avi")], reverse=True)
            for f in files:
                self.rec_list.insert(tk.END, f)

    def play_recording(self):
        selection = self.rec_list.curselection()
        if selection:
            filename = self.rec_list.get(selection[0])
            filepath = os.path.abspath(os.path.join("recordings", filename))
            if sys.platform == "darwin":
                subprocess.call(('open', filepath))
            elif sys.platform == "win32":
                os.startfile(filepath)
            else:
                subprocess.call(('xdg-open', filepath))

if __name__ == "__main__":
    root = tk.Tk()
    app = SecuritySystem(root, "Northampton Warehouse Security System v1.0")
    root.mainloop()