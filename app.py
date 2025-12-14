import os
import numpy as np
import joblib
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash
from io import BytesIO

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

# --- Flask Configuration ---
app = Flask(__name__)
# IMPORTANT: Use a stronger secret key in production
app.config['SECRET_KEY'] = 'your_strong_secret_key_here' 
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# --- Model Configuration (MUST MATCH TRAINING) ---
SAVE_DIR = "models_and_features"
FEATURE_SET_NAME = "ResNet50+VGG16"
# !!! IMPORTANT: REPLACE XX with the actual iteration number from your saved files
BEST_ITERATION = 14 

# Constants for Patching
IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 1024
NUM_H_PATCHES = 8
NUM_V_PATCHES = 8
PATCH_SIZE = IMAGE_HEIGHT // NUM_V_PATCHES 
THETA_CENTERS = [-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5]
PHI_CENTERS = [-78.75, -56.25, -33.75, -11.25, 11.25, 33.75, 56.25, 78.75]

# --- Global Variables for Loaded Pipeline Components ---
VISUAL_ENCODER = None
POS_ENCODER = None
ATTENTION_MODEL = None
SCALER = None
PCA = None
SVR_MODEL = None
POPT_GLOBAL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Architecture Definitions (Necessary for loading state_dict) ---

class FeatureEncoder(nn.Module):
    def __init__(self, model_name, weights=None):
        super().__init__()
        if model_name == 'ResNet50':
            base = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(base.children())[:-1])
            self.out_dim = base.fc.in_features
        elif model_name == 'VGG16':
            base = models.vgg16(weights=weights)
            self.features = nn.Sequential(*list(base.features.children()), nn.AdaptiveAvgPool2d((1, 1)))
            self.out_dim = 512
        else: self.out_dim = 0
            
    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)

class DualVisualFeatureEncoder(nn.Module):
    def __init__(self, encoder1_name='ResNet50', encoder2_name='VGG16', weights=None):
        super().__init__()
        self.encoder1 = FeatureEncoder(encoder1_name, weights=weights) 
        self.encoder2 = FeatureEncoder(encoder2_name, weights=weights)
        self.out_dim = self.encoder1.out_dim + self.encoder2.out_dim
    def forward(self, x):
        f1 = self.encoder1(x)
        f2 = self.encoder2(x)
        return torch.cat([f1, f2], dim=-1)

class PositionalEncoder(nn.Module):
    def __init__(self, in_dim=4, hid=64, out_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, out_dim), nn.ReLU())
        self.out_dim = out_dim
    def forward(self, x):
        return self.mlp(x)

class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim, attention_dim=128):
        super().__init__()
        self.attention_mlp = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))
        self.regressor = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, combined_feats):
        scores = self.attention_mlp(combined_feats)
        weights = F.softmax(scores, dim=1)
        weighted_sum = torch.sum(weights * combined_feats, dim=1)
        return weighted_sum, weights

# --- Preprocessing & Logistic Function ---

eval_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def logistic_func(x, beta1, beta2, beta3, beta4):
    """Sigmoid function used for final mapping."""
    return beta1 * (0.5 - 1.0 / (1 + np.exp(beta2 * (x - beta3)))) + beta4

def extract_grid_patches(img_pil, transform):
    """Extracts patches and positional encodings."""
    img_pil = img_pil.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BICUBIC)
    patches = []
    pos_encs = []
    for r in range(NUM_V_PATCHES):
        for c in range(NUM_H_PATCHES):
            left, top = c * PATCH_SIZE, r * PATCH_SIZE
            patch = img_pil.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
            patches.append(transform(patch))

            theta = torch.deg2rad(torch.tensor(THETA_CENTERS[c], dtype=torch.float32))
            phi = torch.deg2rad(torch.tensor(PHI_CENTERS[r], dtype=torch.float32))
            pos_enc = torch.stack([torch.sin(theta), torch.cos(theta),
                                   torch.sin(phi),    torch.cos(phi)], dim=0)
            pos_encs.append(pos_enc)

    return torch.stack(patches, dim=0), torch.stack(pos_encs, dim=0)

# --- Core Prediction Logic ---

@torch.no_grad()
def predict_quality_score(img_stream, visual_encoder, pos_encoder, attention_model, scaler, pca, svr_model, popt_global):
    """Runs the full pipeline on a PIL image stream."""
    try:
        # Read image from stream
        img_pil = Image.open(img_stream).convert("RGB")
    except Exception as e:
        print(f"Error opening image stream: {e}")
        return None

    patches, pos_encs = extract_grid_patches(img_pil, eval_tf)
    patches = patches.unsqueeze(0).to(DEVICE)
    pos_encs = pos_encs.unsqueeze(0).to(DEVICE)
    B, N = patches.size(0), patches.size(1)

    # 1. Feature Extraction
    img_feats_flat = visual_encoder(patches.view(B*N, 3, 224, 224)) 
    img_feats = img_feats_flat.view(B, N, -1)
    pos_feats = pos_encoder(pos_encs)        
    combined_feats = torch.cat([img_feats, pos_feats], dim=-1)
    
    # 2. Attention Aggregation
    attn_feats, _ = attention_model(combined_feats)
    final_feature = attn_feats.cpu().numpy().flatten().reshape(1, -1) 

    # 3. Regression Pipeline (Sklearn)
    # Global variables are assumed to be loaded
    feature_scaled = SCALER.transform(final_feature)
    feature_pca = PCA.transform(feature_scaled)
    raw_score = SVR_MODEL.predict(feature_pca)[0]
    
    # 4. Logistic Mapping
    final_score = logistic_func(raw_score, *POPT_GLOBAL)
    
    return final_score

# --- Pipeline Loading Function ---

def load_pipeline_components():
    """Loads all necessary models into global scope for single-time loading."""
    global VISUAL_ENCODER, POS_ENCODER, ATTENTION_MODEL, SCALER, PCA, SVR_MODEL, POPT_GLOBAL
    
    base_name = f"{FEATURE_SET_NAME}_iter_{BEST_ITERATION}"
    
    try:
        # Load Sklearn Components
        SCALER = joblib.load(os.path.join(SAVE_DIR, f'{base_name}_scaler.pkl'))
        PCA = joblib.load(os.path.join(SAVE_DIR, f'{base_name}_pca.pkl'))
        SVR_MODEL = joblib.load(os.path.join(SAVE_DIR, f'{base_name}_svr_model.pkl'))
        POPT_GLOBAL = joblib.load(os.path.join(SAVE_DIR, f'{base_name}_popt_global.pkl'))
        
        # Load PyTorch Models
        VISUAL_ENCODER = DualVisualFeatureEncoder(weights=None)
        VISUAL_ENCODER.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'{base_name}_feature_encoder.pt'), map_location=DEVICE))
        VISUAL_ENCODER.to(DEVICE).eval()
        
        POS_ENCODER = PositionalEncoder()
        POS_ENCODER.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'{base_name}_pos_encoder.pt'), map_location=DEVICE))
        POS_ENCODER.to(DEVICE).eval()
        
        combined_feat_dim = VISUAL_ENCODER.out_dim + POS_ENCODER.out_dim
        ATTENTION_MODEL = AttentionAggregator(feature_dim=combined_feat_dim)
        ATTENTION_MODEL.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'{base_name}_attention_agg.pt'), map_location=DEVICE))
        ATTENTION_MODEL.to(DEVICE).eval()

        print("--- Model Pipeline Loaded Successfully ---")
        return True
    except Exception as e:
        print(f"--- FAILED TO LOAD MODEL COMPONENTS: {e} ---")
        return False

# --- Flask Routes ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handles the home page and image upload."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            
            # --- 1. Save File ---
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create a copy of the file stream to save to disk
            file_stream_copy = BytesIO(file.read())
            
            # Save the file to the uploads folder
            with open(filepath, 'wb') as f:
                f.write(file_stream_copy.getvalue())
                
            # --- 2. Run Prediction ---
            try:
                # Pass the stream copy to prediction function
                score = predict_quality_score(
                    file_stream_copy, VISUAL_ENCODER, POS_ENCODER, ATTENTION_MODEL, SCALER, PCA, SVR_MODEL, POPT_GLOBAL
                )
                
                if score is not None:
                    # Pass filename to template for image display
                    return render_template('index.html', 
                                           predicted_score=f"{score:.4f}", 
                                           filename=filename)
                else:
                    flash('Error during prediction pipeline.')
                    # Clean up the file if prediction failed
                    os.remove(filepath)
                    return redirect(request.url)
            
            except Exception as e:
                flash(f'An error occurred during prediction: {e}')
                # Clean up the file if prediction failed
                os.remove(filepath)
                return redirect(request.url)
        else:
            flash('File type not allowed. Use .png, .jpg, or .jpeg.')
            return redirect(request.url)
            
    # GET request: Render the upload form
    return render_template('index.html', predicted_score=None)

# --- Application Startup ---

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load models before running the app
    if load_pipeline_components():
        # Set static URL path to serve files from the 'uploads' folder
        # Flask automatically serves static files from a 'static' folder, 
        # but we need to configure it for 'uploads' too.
        # Alternatively, we rely on the default behavior and access uploads via the public URL path.
        app.run(debug=True)