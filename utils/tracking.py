import numpy as np
import cv2
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

class ReIDTracker:
    def __init__(self, max_disappeared=30, max_distance=0.7):
        self.next_id = 0
        self.tracks = {}  # {track_id: {'bbox': (x1,y1,x2,y2), 'features': [], 'last_seen': frame_count}}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.frame_count = 0
        
    def _extract_features(self, frame, bbox):
        """Extract robust appearance features from player crop"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Handle invalid coordinates
        if x1 >= x2 or y1 >= y2 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return np.zeros(72)  # Return zero features for invalid regions
        
        player_crop = frame[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return np.zeros(72)
        
        features = []
        
        # 1. Color histogram in HSV space (more robust to lighting)
        hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        
        # Normalize and concatenate
        features.extend(cv2.normalize(hist_h, None).flatten())
        features.extend(cv2.normalize(hist_s, None).flatten())
        features.extend(cv2.normalize(hist_v, None).flatten())
        
        # 2. Dominant jersey color
        # Focus on upper half of player (jersey area)
        jersey_region = player_crop[:int(player_crop.shape[0]*0.5), :]
        if jersey_region.size > 0:
            dominant_color = np.mean(jersey_region.reshape(-1, 3), axis=0)
            features.extend(dominant_color / 255.0)
        else:
            features.extend([0, 0, 0])
        
        # 3. Team color identification (crude but effective)
        mean_color = np.mean(player_crop.reshape(-1, 3), axis=0)
        features.extend(mean_color / 255.0)
        
        return np.array(features)
    
    def _match_detections_to_tracks(self, detections, frame):
        """Match using appearance + spatial similarity"""
        if not self.tracks:
            return [], list(range(len(detections)))
        
        # Initialize distance matrix
        distance_matrix = np.ones((len(self.tracks), len(detections))) * 1000
        
        # Get track IDs and detection indices
        track_ids = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, det in enumerate(detections):
                # Extract features for new detection
                det_features = self._extract_features(frame, det[:4])
                
                # Calculate appearance similarity
                app_distance = cosine(track['features'], det_features)
                
                # Calculate spatial distance (center points)
                track_center = np.array([(track['bbox'][0] + track['bbox'][2]) / 2,
                                         (track['bbox'][1] + track['bbox'][3]) / 2])
                det_center = np.array([(det[0] + det[2]) / 2, (det[1] + det[3]) / 2])
                spatial_distance = np.linalg.norm(track_center - det_center) / max(frame.shape[0], frame.shape[1])
                
                # Combined distance (weighted)
                combined_distance = 0.7 * app_distance + 0.3 * spatial_distance
                distance_matrix[i, j] = combined_distance
        
        # Find optimal matching using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for i, j in zip(row_ind, col_ind):
            if distance_matrix[i, j] < self.max_distance:
                matches.append((track_ids[i], j))
                if j in unmatched_detections:
                    unmatched_detections.remove(j)
            else:
                unmatched_tracks.append(track_ids[i])
        
        return matches, unmatched_detections

    def update(self, detections, frame):
        self.frame_count += 1
        
        # Update existing tracks
        active_tracks = []
        
        # Match detections to tracks
        matches, unmatched_detections = self._match_detections_to_tracks(detections, frame)
        
        # Update matched tracks
        for track_id, det_idx in matches:
            det = detections[det_idx]
            features = self._extract_features(frame, det[:4])
            
            # Update track
            self.tracks[track_id]['bbox'] = det[:4]
            self.tracks[track_id]['features'] = features
            self.tracks[track_id]['last_seen'] = self.frame_count
            
            active_tracks.append((*det[:4], track_id))
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            features = self._extract_features(frame, det[:4])
            
            track_id = self.next_id
            self.tracks[track_id] = {
                'bbox': det[:4],
                'features': features,
                'last_seen': self.frame_count
            }
            active_tracks.append((*det[:4], track_id))
            self.next_id += 1
        
        # Remove stale tracks
        stale_tracks = []
        for track_id, track in self.tracks.items():
            if self.frame_count - track['last_seen'] > self.max_disappeared:
                stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            del self.tracks[track_id]
        
        return active_tracks