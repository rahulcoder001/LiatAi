import cv2
import time
from utils.detection import PlayerDetector
from utils.tracking import ReIDTracker
from utils.visualization import draw_tracks

def main():
    # Initialize components
    detector = PlayerDetector("data/best.pt")
    tracker = ReIDTracker(max_disappeared=30, max_distance=0.7)
    
    # Open video
    cap = cv2.VideoCapture("data/15sec_input_720p.mp4")
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect players
        detections = detector.detect(frame)
        
        # Update tracks
        tracks = tracker.update(detections, frame)
        
        # Visualize results
        output_frame = draw_tracks(frame, tracks)
        
        # Write to output
        out.write(output_frame)
        
        # Display processing info
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        cv2.putText(output_frame, f"FPS: {fps:.1f} | Players: {len(tracks)}", 
                   (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Player Tracking", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete! {frame_count} frames processed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()