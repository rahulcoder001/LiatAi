import cv2
import numpy as np

# Distinct colors for player IDs
COLORS = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 0),    # Dark Green
    (0, 0, 128)     # Dark Red
]

def draw_tracks(frame, tracks):
    if len(tracks) == 0:
        return frame
    
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        color = COLORS[int(track_id) % len(COLORS)]
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw ID background for better visibility
        label = f"P-{int(track_id)}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (int(x1), int(y1)-25), (int(x1)+text_width, int(y1)), color, -1)
        
        # Draw player ID
        cv2.putText(frame, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add frame counter
    cv2.putText(frame, f"Players: {len(tracks)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame