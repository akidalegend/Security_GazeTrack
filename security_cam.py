import cv2
import sys
import numpy as np
import datetime
import os
import subprocess
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from gaze_tracking import GazeTracking

class SecuritySystem:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x800")
        
        # --- MODERN COLOR PALETTE ---
        self.colors = {
            "bg": "#0F172A",       # Deep Navy
            "card": "#1E293B",     # Slate
            "text": "#F8FAFC",     # Ghost White
            "dim": "#94A3B8",      # Muted Blue/Grey
            "accent": "#0EA5E9",   # Cyan
            "alert": "#F43F5E",    # Rose/Red
            "success": "#10B981"   # Emerald/Green
        }
        self.window.configure(bg=self.colors["bg"])

        # --- LOGIC INITIALIZATION ---
        print("Initialising AI Core...")
        self.model = YOLO('yolov8n.pt') 
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.gaze = GazeTracking()
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.is_recording = False
        self.out = None
        self.no_motion_frames = 0
        self.recording_cooldown = 30
        self.latest_frame = None

        # Variables
        self.use_roi = tk.BooleanVar(value=True)
        self.use_gaze = tk.BooleanVar(value=False)
        self.use_low_light = tk.BooleanVar(value=True)
        self.use_distortion_correction = tk.BooleanVar(value=True)
        self.use_auto_roi = tk.BooleanVar(value=False)  # NEW: Auto ROI detection
        
        # Auto ROI variables
        self.roi_coords = None
        self.roi_learning_frames = 0
        self.roi_max_learning = 100  # Learn ROI over first 100 frames

        # Camera calibration parameters (simulated for typical webcam)
        self.camera_matrix = np.array([
            [1280, 0, 640],      # fx, 0, cx
            [0, 1280, 360],      # 0, fy, cy  
            [0, 0, 1]            # 0, 0, 1
        ], dtype=np.float32)
        
        # Distortion coefficients: k1, k2, p1, p2, k3
        self.distortion_coeffs = np.array([-0.2, 0.1, 0.001, 0.001, -0.05], dtype=np.float32)

        # --- UI LAYOUT ---
        self.header = tk.Frame(self.window, bg=self.colors["card"], height=70)
        self.header.pack(fill=tk.X, side=tk.TOP)
        self.header.pack_propagate(False)

        tk.Label(self.header, text="SATORU GOJO SIX EYES SYSTEM", font=("Futura", 24, "bold"), 
                 bg=self.colors["card"], fg=self.colors["accent"]).pack(side=tk.LEFT, padx=30)
        
        self.status_indicator = tk.Label(self.header, text="● SYSTEM ONLINE", font=("Helvetica", 12, "bold"), 
                                        bg=self.colors["card"], fg=self.colors["success"])
        self.status_indicator.pack(side=tk.RIGHT, padx=30)

        self.workspace = tk.Frame(self.window, bg=self.colors["bg"], pady=20, padx=20)
        self.workspace.pack(fill=tk.BOTH, expand=True)

        self.video_card = tk.Frame(self.workspace, bg="black", bd=0)
        self.video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.video_card, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.sidebar = tk.Frame(self.workspace, bg=self.colors["bg"], width=350)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        self.sidebar.pack_propagate(False)

        self.control_panel = tk.LabelFrame(self.sidebar, text=" CONFIGURATION ", bg=self.colors["card"], 
                                          fg=self.colors["dim"], font=("Arial", 10, "bold"), padx=15, pady=15)
        self.control_panel.pack(fill=tk.X, pady=(0, 20))

        tk.Checkbutton(self.control_panel, text="Enable Perimeter Detection", variable=self.use_roi, 
                       bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
                       activebackground=self.colors["card"], font=("Arial", 11)).pack(anchor="w")
        
        tk.Checkbutton(self.control_panel, text="Biometric Gaze Tracking", variable=self.use_gaze, 
                       bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
                       activebackground=self.colors["card"], font=("Arial", 11), command=self.toggle_gaze).pack(anchor="w", pady=(10, 0))

        tk.Checkbutton(self.control_panel, text="Low Light Enhancement", variable=self.use_low_light, 
                       bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
                       activebackground=self.colors["card"], font=("Arial", 11), command=self.toggle_low_light).pack(anchor="w", pady=(10, 0))

        # NEW: Distortion correction toggle
        tk.Checkbutton(self.control_panel, text="Lens Distortion Correction", variable=self.use_distortion_correction, 
                       bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
                       activebackground=self.colors["card"], font=("Arial", 11), command=self.toggle_distortion_correction).pack(anchor="w", pady=(10, 0))

        # Add Auto ROI checkbox
        tk.Checkbutton(self.control_panel, text="Automatic ROI Detection", variable=self.use_auto_roi, 
                       bg=self.colors["card"], fg=self.colors["text"], selectcolor=self.colors["bg"],
                       activebackground=self.colors["card"], font=("Arial", 11), command=self.toggle_auto_roi).pack(anchor="w", pady=(10, 0))

        tk.Label(self.control_panel, text="MOTION SENSITIVITY", bg=self.colors["card"], 
                 fg=self.colors["dim"], font=("Arial", 8, "bold")).pack(anchor="w", pady=(20, 5))
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TScale", background=self.colors["card"])
        
        self.sensitivity = ttk.Scale(self.control_panel, from_=500, to=5000, orient=tk.HORIZONTAL)
        self.sensitivity.set(1000)
        self.sensitivity.pack(fill=tk.X)

        self.tabs = ttk.Notebook(self.sidebar)
        self.tabs.pack(fill=tk.BOTH, expand=True)

        self.tab_log = tk.Frame(self.tabs, bg=self.colors["card"])
        self.tabs.add(self.tab_log, text=" ACTIVITY ")
        self.log_text = tk.Text(self.tab_log, bg="#000", fg=self.colors["success"], 
                                font=("Courier", 10), bd=0, padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.tab_rec = tk.Frame(self.tabs, bg=self.colors["card"])
        self.tabs.add(self.tab_rec, text=" ARCHIVE ")
        self.rec_list = tk.Listbox(self.tab_rec, bg="#000", fg="white", bd=0, 
                                   selectbackground=self.colors["accent"], font=("Arial", 10))
        self.rec_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Button(self.tab_rec, text="▶ VIEW FOOTAGE", command=self.play_recording, 
                  bg="#1E293B", fg="#F8FAFC", relief="flat", font=("Arial", 10, "bold"),
                  activebackground="#334155", activeforeground="#F8FAFC", 
                  borderwidth=0, highlightthickness=0).pack(fill=tk.X, padx=5, pady=5)

        self.btn_quit = tk.Button(self.sidebar, text="TERMINATE SYSTEM", command=self.quit_app, 
                                  bg="#1E293B", fg="#F8FAFC", relief="flat", 
                                  font=("Arial", 11, "bold"), pady=12,
                                  activebackground="#334155", activeforeground="#F8FAFC",
                                  borderwidth=0, highlightthickness=0)
        self.btn_quit.pack(side=tk.BOTTOM, fill=tk.X)

        self.refresh_recordings()
        self.update_loop()

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
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))
        self.is_recording = True
        self.status_indicator.config(text="● RECORDING", fg=self.colors["alert"])
        self.log_message(f"TRIGGER: Recording started")

    def stop_recording(self):
        if self.is_recording:
            self.out.release()
            self.is_recording = False
            self.status_indicator.config(text="● SYSTEM ONLINE", fg=self.colors["success"])
            self.log_message("STATUS: Recording saved")
            self.refresh_recordings()

    def correct_distortion(self, frame):
        """Apply camera distortion correction using calibration parameters"""
        return cv2.undistort(frame, self.camera_matrix, self.distortion_coeffs)

    def enhance_low_light(self, frame):
        """Apply low light enhancement using histogram equalization"""
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        enhanced_frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return enhanced_frame

    def detect_auto_roi(self, frame):
        """Automatically detect ROI based on scene analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Edge-based ROI (detect doorways, windows, etc.)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours of significant structures
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        significant_contours = []
        h, w = frame.shape[:2]
        min_area = (w * h) * 0.02  # At least 2% of frame
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                # Check if it's a reasonable shape for a doorway/window
                aspect_ratio = ch / cw
                if 0.5 < aspect_ratio < 3.0:  # Not too wide or too tall
                    significant_contours.append((x, y, x+cw, y+ch))
        
        if significant_contours:
            # Find the most central significant structure
            center_x, center_y = w//2, h//2
            closest_dist = float('inf')
            best_roi = None
            
            for x1, y1, x2, y2 in significant_contours:
                roi_center_x = (x1 + x2) // 2
                roi_center_y = (y1 + y2) // 2
                dist = np.sqrt((roi_center_x - center_x)**2 + (roi_center_y - center_y)**2)
                
                if dist < closest_dist:
                    closest_dist = dist
                    best_roi = (x1, y1, x2, y2)
            
            return best_roi
        
        # Method 2: Motion-based ROI learning
        return self.learn_roi_from_motion(frame)

    def learn_roi_from_motion(self, frame):
        """Learn ROI from accumulated motion patterns"""
        if self.roi_learning_frames < self.roi_max_learning:
            # Apply background subtraction
            mask = self.fgbg.apply(frame)
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            
            # Find motion areas
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get bounding box of all motion
                all_points = np.vstack([contour for contour in contours])
                x, y, w, h = cv2.boundingRect(all_points)
                
                # Expand ROI slightly for better coverage
                padding = 50
                h_frame, w_frame = frame.shape[:2]
                
                roi_x1 = max(0, x - padding)
                roi_y1 = max(0, y - padding)
                roi_x2 = min(w_frame, x + w + padding)
                roi_y2 = min(h_frame, y + h + padding)
                
                if self.roi_coords is None:
                    self.roi_coords = [roi_x1, roi_y1, roi_x2, roi_y2]
                else:
                    # Gradually expand ROI to encompass all detected motion
                    self.roi_coords[0] = min(self.roi_coords[0], roi_x1)
                    self.roi_coords[1] = min(self.roi_coords[1], roi_y1)
                    self.roi_coords[2] = max(self.roi_coords[2], roi_x2)
                    self.roi_coords[3] = max(self.roi_coords[3], roi_y2)
            
            self.roi_learning_frames += 1
            
            if self.roi_learning_frames == self.roi_max_learning:
                self.log_message("AUTO-ROI: Learning complete")
        
        return tuple(self.roi_coords) if self.roi_coords else None

    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            self.latest_frame = frame
            
            # Apply distortion correction if enabled
            if self.use_distortion_correction.get():
                frame = self.correct_distortion(frame)
            
            # Apply low light enhancement if enabled
            if self.use_low_light.get():
                frame = self.enhance_low_light(frame)
            
            # Initialize auto_roi variable
            auto_roi = None
            
            # CREATE ROI MASK FIRST (before background subtraction)
            roi_frame = frame.copy()  # Work with a copy for ROI processing
            yolo_frame = frame.copy()  # Separate frame for YOLO processing
            
            if self.use_roi.get():
                if self.use_auto_roi.get():
                    # Auto-detect ROI
                    auto_roi = self.detect_auto_roi(frame)
                    if auto_roi:
                        roi_x1, roi_y1, roi_x2, roi_y2 = auto_roi
                    else:
                        # Fallback to default ROI
                        h, w = frame.shape[:2]
                        roi_x1, roi_y1 = int(w*0.1), int(h*0.3)
                        roi_x2, roi_y2 = int(w*0.9), int(h*0.9)
                else:
                    # Manual ROI (existing behavior)
                    h, w = frame.shape[:2]
                    roi_x1, roi_y1 = int(w*0.1), int(h*0.3)
                    roi_x2, roi_y2 = int(w*0.9), int(h*0.9)
                
                # Apply ROI mask to BOTH motion detection AND YOLO frames
                roi_frame = np.zeros_like(frame)
                roi_frame[roi_y1:roi_y2, roi_x1:roi_x2] = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Mask YOLO frame as well
                yolo_frame = np.zeros_like(frame)
                yolo_frame[roi_y1:roi_y2, roi_x1:roi_x2] = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Apply background subtraction to the ROI-masked frame
            mask = self.fgbg.apply(roi_frame)
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = any(cv2.contourArea(c) > self.sensitivity.get() for c in contours)

            if self.use_gaze.get():
                self.gaze.refresh(frame)
            
            # Run YOLO on the ROI-masked frame (or original if ROI disabled)
            results = self.model(yolo_frame, verbose=False)
            annotated_frame = results[0].plot()
            
            # Draw Auto ROI if enabled (fixed condition)
            if self.use_roi.get() and self.use_auto_roi.get() and auto_roi is not None:
                roi_x1, roi_y1, roi_x2, roi_y2 = auto_roi
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 165, 0), 2)
                cv2.putText(annotated_frame, "AUTO-ROI", (roi_x1, roi_y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Draw manual ROI if enabled and auto ROI is disabled
            elif self.use_roi.get() and not self.use_auto_roi.get():
                h, w = annotated_frame.shape[:2]
                roi_x1, roi_y1 = int(w*0.1), int(h*0.3)
                roi_x2, roi_y2 = int(w*0.9), int(h*0.9)
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "PERIMETER ZONE", (roi_x1, roi_y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if self.use_gaze.get():
                annotated_frame = self.draw_gaze_overlay(annotated_frame)

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
                self.out.write(annotated_frame)

            cv_w, cv_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cv_w > 1:
                annotated_frame = cv2.resize(annotated_frame, (cv_w, cv_h))

            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.image = img

        self.window.after(10, self.update_loop)

    def toggle_gaze(self):
        state = "ENABLED" if self.use_gaze.get() else "DISABLED"
        self.log_message(f"SYSTEM: Gaze tracking {state}")

    def toggle_low_light(self):
        state = "ENABLED" if self.use_low_light.get() else "DISABLED"
        self.log_message(f"SYSTEM: Low light enhancement {state}")

    def toggle_distortion_correction(self):
        state = "ENABLED" if self.use_distortion_correction.get() else "DISABLED"
        self.log_message(f"SYSTEM: Lens distortion correction {state}")

    def toggle_auto_roi(self):
        if self.use_auto_roi.get():
            self.roi_coords = None
            self.roi_learning_frames = 0
            self.log_message("AUTO-ROI: Learning mode activated")
        else:
            self.log_message("AUTO-ROI: Manual mode restored")

    def draw_gaze_overlay(self, frame):
        h, w = frame.shape[:2]
        l, r = self.gaze.pupil_left_coords(), self.gaze.pupil_right_coords()
        for p in [l, r]:
            if p: cv2.circle(frame, p, 5, (0, 255, 0), -1)
        return frame

    def refresh_recordings(self):
        self.rec_list.delete(0, tk.END)
        if os.path.exists("recordings"):
            for f in sorted([f for f in os.listdir("recordings") if f.endswith(".avi")], reverse=True):
                self.rec_list.insert(tk.END, f)

    def play_recording(self):
        selection = self.rec_list.curselection()
        if selection:
            path = os.path.abspath(os.path.join("recordings", self.rec_list.get(selection[0])))
            subprocess.call(('open', path))

    def quit_app(self):
        self.stop_recording()
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SecuritySystem(root, "SATORU GOJO SIX EYES SYSTEM ")
    root.mainloop()