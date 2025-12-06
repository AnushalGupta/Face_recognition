import customtkinter as ctk
import cv2
import os
import numpy as np
import csv
import datetime
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# ⚙️ CONFIGURATION (CHANGE NAME HERE)
# ==========================================
APP_NAME = "Face_Recognition_APP"  # <--- CHANGE THIS TO YOUR APP NAME
DB_PATH = "faces"
LOG_FILE = "access_log.csv"
FRAME_SKIP = 5           # Run AI every 5th frame
RESIZE_FACTOR = 0.5      # Process at 50% resolution for speed
MOTION_THRESHOLD = 1000  # Sensitivity of motion sensor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==========================================
# 🧠 LITE AI ENGINE (The Brain)
# ==========================================
class SecurityEngine:
    def __init__(self):
        print("[INIT] Loading Mobile AI Model...")
        # Uses 'buffalo_s' (Small) for maximum FPS on low-end devices
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        
        self.db = {}
        self.load_db()
        self.prev_gray = None

    def load_db(self):
        if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)
        self.db = {}
        for f in os.listdir(DB_PATH):
            if f.endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(DB_PATH, f))
                if img is not None:
                    faces = self.app.get(img)
                    if faces:
                        name = os.path.splitext(f)[0]
                        self.db[name] = faces[0].embedding
        print(f"Database Loaded: {len(self.db)} users.")

    def register_user(self, frame, name):
        # Save full resolution image for better quality reference
        path = os.path.join(DB_PATH, f"{name}.jpg")
        cv2.imwrite(path, frame)
        self.load_db() # Reload memory

    def detect_motion(self, frame):
        """Cheap/Fast way to see if we need to run AI"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        change = np.count_nonzero(thresh)
        self.prev_gray = gray
        return change > MOTION_THRESHOLD

    def process_frame(self, frame):
        """Optimized Inference"""
        # Downscale for AI speed
        small = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        faces = self.app.get(small)
        results = []
        
        for face in faces:
            # Upscale box to match original video
            bbox = (face.bbox / RESIZE_FACTOR).astype(int)
            embedding = face.embedding
            
            best_name = "Unknown"
            best_score = 0.0
            
            for name, db_emb in self.db.items():
                score = cosine_similarity([embedding], [db_emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_name = name
            
            final_name = best_name if best_score > 0.45 else "Unknown"
            color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)
            
            results.append((bbox, final_name, color))
            
        return results

# ==========================================
# 🖥️ GUI (The Interface)
# ==========================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1100x700")
        self.title(APP_NAME) # <--- Uses the name from config

        # Grid Layout (2 Columns: Sidebar | Video)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR (Left) ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.lbl_logo = ctk.CTkLabel(self.sidebar, text=APP_NAME, font=ctk.CTkFont(size=20, weight="bold"))
        self.lbl_logo.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_reg = ctk.CTkButton(self.sidebar, text="Add New User", command=self.event_register)
        self.btn_reg.grid(row=1, column=0, padx=20, pady=10)

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="STATUS: Standby", font=("Arial", 14))
        self.lbl_status.grid(row=2, column=0, padx=20, pady=20)

        self.btn_quit = ctk.CTkButton(self.sidebar, text="Shut Down", fg_color="red", hover_color="darkred", command=self.destroy)
        self.btn_quit.grid(row=5, column=0, padx=20, pady=(300, 20))

        # --- VIDEO AREA (Right) ---
        self.video_frame = ctk.CTkLabel(self, text="")
        self.video_frame.grid(row=0, column=1, padx=20, pady=20)

        # --- SYSTEM START ---
        self.engine = SecurityEngine()
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Use Index 1 for Iriun
        
        self.frame_count = 0
        self.last_results = []
        self.update_feed()

    def event_register(self):
        # 1. Ask for name
        dialog = ctk.CTkInputDialog(text="Enter Name of Person:", title="Registration")
        name = dialog.get_input()
        
        if name:
            # 2. Capture current frame instantly
            ret, frame = self.cap.read()
            if ret:
                self.engine.register_user(frame, name)
                self.lbl_status.configure(text=f"Saved: {name}", text_color="green")

    def update_feed(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            
            # --- INTELLIGENT LOGIC ---
            # Run AI only every N frames AND if motion is detected
            if self.frame_count % FRAME_SKIP == 0:
                is_moving = self.engine.detect_motion(frame)
                
                if is_moving:
                    self.lbl_status.configure(text="STATUS: Scanning...", text_color="orange")
                    self.last_results = self.engine.process_frame(frame)
                else:
                    self.lbl_status.configure(text="STATUS: Standby", text_color="gray")
                    # Clear boxes if no motion for a while
                    if self.frame_count % (FRAME_SKIP * 4) == 0:
                         self.last_results = []

            # --- DRAW RESULTS ---
            for res in self.last_results:
                bbox, name, color = res
                x1, y1, x2, y2 = bbox
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Name Tag Background
                (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1-30), (x1+w, y1), color, -1)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Update Sidebar if match found
                if name != "Unknown":
                    self.lbl_status.configure(text=f"ACCESS: {name}", text_color="cyan")

            # --- RENDER TO GUI ---
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            # Resize image to fit window nicely
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
            
            self.video_frame.configure(image=ctk_img)
            self.video_frame.image = ctk_img

        self.after(10, self.update_feed)

if __name__ == "__main__":
    app = App()
    app.mainloop()
