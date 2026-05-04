import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def _check_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None

def get_video_info(video_source):
    cv2 = _check_cv2()
    if not cv2: return {"Error": "Install opencv-python to read video"}
    """Obtain basic info using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            return {"Error": "Cannot open video."}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else 0
        cap.release()
        
        return {
            "Resolution": f"{width}x{height}",
            "FPS": round(fps, 2),
            "Total Frames": frame_count,
            "Duration (s)": round(duration, 2)
        }
    except Exception as e:
        return {"Error": str(e)}

def get_video_frame_fig(video_source, time_sec):
    cv2 = _check_cv2()
    if not cv2:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "opencv required", ha="center"); return fig
    try:
        cap = cv2.VideoCapture(video_source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_no = int(time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#f8fafc")
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)
            ax.set_title(f"Frame at {time_sec}s")
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, "Frame not found", ha="center")
            ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, str(e), ha="center")
        return fig

def get_video_scene_cuts_fig(video_source, max_frames=500):
    cv2 = _check_cv2()
    if not cv2: return None
    """Calculate frame-to-frame diff to find scene cuts."""
    try:
        cap = cv2.VideoCapture(video_source)
        diffs = []
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return None
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret or count > max_frames:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            diffs.append(np.mean(diff))
            prev_gray = gray
            count += 1
            
        cap.release()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#f8fafc")
        ax.plot(diffs, color='teal')
        ax.set_title("Frame-to-Frame Absolute Difference (Scene Cuts)")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Mean Pixel Diff")
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, str(e), ha="center")
        return fig

def get_video_color_timeline_fig(video_source, num_samples=50):
    cv2 = _check_cv2()
    if not cv2: return None
    try:
        cap = cv2.VideoCapture(video_source)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return None
        
        step = max(1, frame_count // num_samples)
        colors = []
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mean_color = frame_rgb.mean(axis=(0,1))
            colors.append(mean_color)
        cap.release()
        
        colors = np.array(colors) / 255.0
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.patch.set_facecolor("#f8fafc")
        
        for idx, color in enumerate(colors):
            ax.axvspan(idx, idx+1, color=color)
            
        ax.set_xlim(0, len(colors))
        ax.set_yticks([])
        ax.set_xlabel("Sampled Intervals")
        ax.set_title("Video Color Timeline")
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def get_video_motion_history_fig(video_source, max_frames=100):
    cv2 = _check_cv2()
    if not cv2: return None
    try:
        cap = cv2.VideoCapture(video_source)
        ret, frame = cap.read()
        if not ret: return None
        
        mhi = np.zeros(frame.shape[:2], np.float32)
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        count = 0
        timestamp = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            _, diff_thresh = cv2.threshold(diff, 30, 1, cv2.THRESH_BINARY)
            
            timestamp += 1
            cv2.motempl.updateMotionHistory(diff_thresh, mhi, timestamp, 30)
            
            prev_gray = gray
            count += 1
            
        cap.release()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#f8fafc")
        ax.imshow(mhi, cmap='magma')
        ax.set_title("Motion History Image")
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        return None
