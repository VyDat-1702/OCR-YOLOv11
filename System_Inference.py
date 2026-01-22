import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from collections import deque, defaultdict
from scipy.optimize import linear_sum_assignment

class YOLOv11Detector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                boxes.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
        return boxes

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3):
        super(CRNN, self).__init__()
        backbone = timm.create_model("resnet34", in_chans=1, pretrained=False)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)
        self.mapSeq = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(512, hidden_size, n_layers, bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2))
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)
        return x

class OCRRecognizer:
    def __init__(self, model_path, vocab_size=37, hidden_size=256, n_layers=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CRNN(vocab_size, hidden_size, n_layers).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        chars = "0123456789abcdefghijklmnopqrstuvwxyz-"
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.transform = transforms.Compose([
            transforms.Resize((100, 420)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def decode(self, encoded_sequences, blank_char="-"):
        decoded_sequences = []
        for seq in encoded_sequences:
            decoded_label = []
            prev_char = None
            for token in seq:
                if token != 0:
                    char = self.idx_to_char[token.item()]
                    if char != blank_char:
                        if char != prev_char or prev_char == blank_char:
                            decoded_label.append(char)
                    prev_char = char
            decoded_sequences.append("".join(decoded_label))
        return decoded_sequences
    
    def recognize(self, img_crop):
        if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
            return ""
        img_pil = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img_tensor)
        pred_text = self.decode(logits.permute(1, 0, 2).argmax(2))[0]
        return pred_text

class TrackedObject:
    def __init__(self, track_id, bbox, text="", max_history=10):
        self.track_id = track_id
        self.bbox_history = deque(maxlen=max_history)  
        self.text_history = deque(maxlen=max_history)  
        self.bbox_history.append(bbox)
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.stable_text = text
        
    def update(self, bbox, text=""):
        self.bbox_history.append(bbox)
        if text:
            self.text_history.append(text)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        
    def get_smoothed_bbox(self, alpha=0.7):
        if len(self.bbox_history) == 0:
            return None
        
        if len(self.bbox_history) == 1:
            return self.bbox_history[-1]
        
        # Exponential moving average
        smoothed = np.array(self.bbox_history[-1][:4], dtype=float)
        for i in range(len(self.bbox_history) - 2, max(-1, len(self.bbox_history) - 6), -1):
            bbox = np.array(self.bbox_history[i][:4], dtype=float)
            smoothed = alpha * smoothed + (1 - alpha) * bbox
            
        return [int(smoothed[0]), int(smoothed[1]), int(smoothed[2]), int(smoothed[3]), 
                self.bbox_history[-1][4]] 
    def get_stable_text(self, min_votes=3):
        if len(self.text_history) < min_votes:
            return self.stable_text if self.stable_text else ""
        
        from collections import Counter
        text_counter = Counter(self.text_history)
        most_common_text, votes = text_counter.most_common(1)[0]
        
        if votes >= min_votes:
            self.stable_text = most_common_text
            
        return self.stable_text
    
    def predict(self):
        self.time_since_update += 1
        self.age += 1
        
        if len(self.bbox_history) >= 2:
            last = np.array(self.bbox_history[-1][:4])
            prev = np.array(self.bbox_history[-2][:4])
            velocity = last - prev
            predicted = last + velocity
            return predicted.astype(int).tolist() + [self.bbox_history[-1][4]]
        return self.bbox_history[-1]

class ObjectTracker:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_counter = 0
        
    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        x1_2, y1_2, x2_2, y2_2 = box2[:4]
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections, ocr_texts=None):

        if ocr_texts is None:
            ocr_texts = [""] * len(detections)
            
        for track in self.tracks:
            track.predict()
        
        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            
            for t, track in enumerate(self.tracks):
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = self.calculate_iou(track.bbox_history[-1], det)
            
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
            
            unmatched_tracks = []
            unmatched_detections = list(range(len(detections)))
            matched_pairs = []
            
            for t, d in matched_indices:
                if iou_matrix[t, d] < self.iou_threshold:
                    unmatched_tracks.append(t)
                    continue
                matched_pairs.append((t, d))
                if d in unmatched_detections:
                    unmatched_detections.remove(d)
            
            for t, d in matched_pairs:
                self.tracks[t].update(detections[d], ocr_texts[d])
            
            for t in range(len(self.tracks)):
                if t not in [pair[0] for pair in matched_pairs]:
                    unmatched_tracks.append(t)
                    
        else:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(detections)))
            matched_pairs = []
        
        for d in unmatched_detections:
            new_track = TrackedObject(self.track_id_counter, detections[d], ocr_texts[d])
            self.tracks.append(new_track)
            self.track_id_counter += 1
        
        self.tracks = [t for i, t in enumerate(self.tracks) 
                      if t.time_since_update < self.max_age]
        
        return [t for t in self.tracks if t.hits >= self.min_hits or t.age < 5]

def process_video(video_path, output_path, yolo_path, ocr_path, 
                 smooth_alpha=0.7, min_votes=3, detect_every_n_frames=2):

    print(f"Loading models...")
    detector = YOLOv11Detector(yolo_path)
    recognizer = OCRRecognizer(ocr_path)
    tracker = ObjectTracker(max_age=10, min_hits=2, iou_threshold=0.3)
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, Total frames: {total_frames}")
    print(f"Smoothing alpha: {smooth_alpha}, Min votes: {min_votes}")
    print(f"Detection every {detect_every_n_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Trying alternative codec...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Cannot create output video")
            cap.release()
            return
    
    print(f"Processing video...")
    frame_count = 0
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % detect_every_n_frames == 1:
                boxes = detector.detect(frame)
                ocr_texts = []
                
                for box in boxes:
                    x1, y1, x2, y2, conf = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        text = recognizer.recognize(crop)
                        ocr_texts.append(text)
                    else:
                        ocr_texts.append("")
                
                tracked_objects = tracker.update(boxes, ocr_texts)
            else:
                tracked_objects = tracker.update([], [])
            
            for track in tracked_objects:
                smoothed_box = track.get_smoothed_bbox(alpha=smooth_alpha)
                if smoothed_box is None:
                    continue
                    
                x1, y1, x2, y2, conf = smoothed_box
                
                stable_text = track.get_stable_text(min_votes=min_votes)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
                cv2.putText(frame, stable_text, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                info_text = f"ID:{track.track_id} {conf:.2f}"
                cv2.putText(frame, info_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            out.write(frame)
            pbar.update(1)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    video_input = "video/input_video.mp4"
    video_output = "vide/demo_video.mp4"
    yolo_model = "model/best.pt"
    ocr_model = "model/ocr_crnn.pt"
    

    process_video(video_input, video_output, yolo_model, ocr_model,
                 smooth_alpha=0.7, min_votes=3, detect_every_n_frames=2)