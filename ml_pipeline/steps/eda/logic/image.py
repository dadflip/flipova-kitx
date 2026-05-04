import io
import base64
import numpy as np
import matplotlib.pyplot as plt

def get_image_info(img) -> dict:
    return {
        "format": getattr(img, 'format', 'Unknown'),
        "size": getattr(img, 'size', 'Unknown'),
        "mode": getattr(img, 'mode', 'Unknown')
    }

def get_image_preview_b64(img, max_width=400) -> str:
    if not hasattr(img, "size"):
        return ""
    scale = min(max_width / max(img.size[0], 1), 1)
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    buf = io.BytesIO()
    preview = img if img.mode == "RGB" else img.convert("RGB")
    preview.resize(new_size).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def get_image_color_histogram_fig(img) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#f8fafc")
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    for i, color in enumerate(("r", "g", "b")):
        hist, _ = np.histogram(arr[:, :, i].ravel(), bins=256, range=[0, 256])
        ax.plot(hist, color=color, alpha=0.8)
        ax.fill_between(range(256), hist, color=color, alpha=0.3)
    ax.set_title("Color Histogram")
    ax.set_xlim([0, 256])
    plt.tight_layout()
    return fig

def get_image_channels_fig(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8fafc")
    colors = ['Reds', 'Greens', 'Blues']
    titles = ['Red Channel', 'Green Channel', 'Blue Channel']
    for i in range(3):
        im = np.zeros_like(arr)
        im[:,:,i] = arr[:,:,i]
        axes[i].imshow(im)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    return fig

def get_image_edges_fig(img):
    try:
        from scipy import ndimage
        if img.mode != "L":
            img_gray = img.convert("L")
        else:
            img_gray = img
        arr = np.array(img_gray, dtype=np.int32)
        dx = ndimage.sobel(arr, 0)
        dy = ndimage.sobel(arr, 1)
        mag = np.hypot(dx, dy)
        mag *= 255.0 / np.max(mag)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#f8fafc")
        ax.imshow(mag, cmap='gray')
        ax.set_title("Edge Detection (Sobel)")
        ax.axis('off')
        plt.tight_layout()
        return fig
    except ImportError:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "pip install scipy requis pour la détection de contours.", ha="center")
        return fig

def get_image_dominant_colors_fig(img, k=5):
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "pip install scikit-learn requis pour les couleurs dominantes.", ha="center")
        return fig
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    arr = arr.reshape((-1, 3))
    # Subsample for speed
    if len(arr) > 10000:
        arr = arr[np.random.choice(len(arr), 10000, replace=False)]
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(arr)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / len(labels)
    # Sort by percentage
    indices = np.argsort(percentages)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor("#f8fafc")
    start = 0
    for i in indices:
        end = start + percentages[i]
        ax.barh(0, percentages[i], left=start, color=f'#{colors[i][0]:02x}{colors[i][1]:02x}{colors[i][2]:02x}')
        start = end
    ax.set_title(f"Top {k} Dominant Colors")
    ax.axis('off')
    plt.tight_layout()
    return fig

def get_image_filters_fig(img):
    try:
        from PIL import ImageFilter
    except ImportError:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "Pillow ImageFilter requis.", ha="center")
        return fig
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#f8fafc")
    
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=5))
    axes[1].imshow(img_blur)
    axes[1].set_title("Gaussian Blur")
    axes[1].axis('off')
    
    img_contour = img.filter(ImageFilter.CONTOUR)
    axes[2].imshow(img_contour)
    axes[2].set_title("Contour Filter")
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

