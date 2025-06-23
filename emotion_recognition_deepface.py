import cv2
from deepface import DeepFace
import numpy as np
from typing import Dict, List, Tuple
import time
from collections import defaultdict
import json

class MultipleEmotionRecognizer:
    def __init__(self):
        """Initialize the emotion recognizer with improved settings"""
        # Emotion to color mapping with better colors
        self.emotion_colors = {
            'happy': (0, 255, 0),         # Green
            'sad': (255, 100, 100),       # Light Blue
            'angry': (0, 0, 255),         # Red
            'fear': (128, 0, 128),        # Purple
            'surprise': (0, 255, 255),    # Yellow
            'disgust': (0, 128, 128),     # Dark Yellow
            'neutral': (200, 200, 200)    # Light Gray
        }
        
        # Face tracking for better stability
        self.face_tracker = {}
        self.next_face_id = 0
        self.face_timeout = 30  # frames
        
        # Statistics tracking
        self.emotion_stats = defaultdict(lambda: defaultdict(int))
        self.frame_count = 0
        self.detection_count = 0
        
        # Performance settings
        self.analysis_interval = 3  # Analyze every 3 frames for better performance
        self.last_analysis_frame = 0
        self.cached_results = []
        
    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """Get color for emotion with fallback"""
        return self.emotion_colors.get(emotion.lower(), (255, 255, 255))
    
    def calculate_face_distance(self, face1: Dict, face2: Dict) -> float:
        """Calculate distance between two face regions for tracking"""
        x1, y1 = face1['x'] + face1['w']//2, face1['y'] + face1['h']//2
        x2, y2 = face2['x'] + face2['w']//2, face2['y'] + face2['h']//2
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    def track_faces(self, current_faces: List[Dict]) -> List[Dict]:
        """Track faces across frames for better stability"""
        tracked_faces = []
        
        # Update existing face tracker
        for face_id in list(self.face_tracker.keys()):
            self.face_tracker[face_id]['age'] += 1
            if self.face_tracker[face_id]['age'] > self.face_timeout:
                del self.face_tracker[face_id]
        
        # Match current faces with tracked faces
        for face in current_faces:
            best_match_id = None
            best_distance = float('inf')
            
            for face_id, tracked_face in self.face_tracker.items():
                distance = self.calculate_face_distance(face['region'], tracked_face['region'])
                if distance < 100 and distance < best_distance:  # Threshold for same face
                    best_distance = distance
                    best_match_id = face_id
            
            if best_match_id is not None:
                # Update existing tracked face
                self.face_tracker[best_match_id]['region'] = face['region']
                self.face_tracker[best_match_id]['age'] = 0
                self.face_tracker[best_match_id]['emotions'] = face['emotion']
                self.face_tracker[best_match_id]['dominant_emotion'] = face['dominant_emotion']
                face['face_id'] = best_match_id
            else:
                # Create new tracked face
                face['face_id'] = self.next_face_id
                self.face_tracker[self.next_face_id] = {
                    'region': face['region'],
                    'emotions': face['emotion'],
                    'dominant_emotion': face['dominant_emotion'],
                    'age': 0
                }
                self.next_face_id += 1
            
            tracked_faces.append(face)
        
        return tracked_faces
    
    def draw_emotion_bars(self, frame: np.ndarray, emotions: Dict[str, float], 
                         position: Tuple[int, int] = (10, 50)) -> None:
        """Draw emotion confidence bars"""
        start_x, start_y = position
        bar_height = 15
        bar_width = 150
        
        # Sort emotions by confidence
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, score) in enumerate(sorted_emotions):
            y = start_y + i * (bar_height + 3)
            
            # Check if bars would go outside frame
            if y + bar_height > frame.shape[0] - 20:
                break
            
            filled_width = int(bar_width * score / 100.0)
            color = self.get_emotion_color(emotion)
            
            # Background bar
            cv2.rectangle(frame, (start_x, y), (start_x + bar_width, y + bar_height), 
                         (40, 40, 40), -1)
            
            # Filled bar
            cv2.rectangle(frame, (start_x, y), (start_x + filled_width, y + bar_height), 
                         color, -1)
            
            # Border
            cv2.rectangle(frame, (start_x, y), (start_x + bar_width, y + bar_height), 
                         (100, 100, 100), 1)
            
            # Text
            text = f"{emotion}: {score:.1f}%"
            cv2.putText(frame, text, (start_x + bar_width + 10, y + bar_height - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_face_info(self, frame: np.ndarray, face_data: Dict, face_index: int) -> None:
        """Draw face rectangle and emotion information"""
        region = face_data['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        
        # Face rectangle with unique color
        face_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                      (255, 0, 255), (0, 255, 255)]
        face_color = face_colors[face_index % len(face_colors)]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)
        
        # Face ID and dominant emotion
        dominant_emotion = face_data['dominant_emotion']
        face_id = face_data.get('face_id', face_index)
        
        label = f"Face {face_id}: {dominant_emotion.upper()}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Label background
        cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), face_color, -1)
        
        # Label text
        cv2.putText(frame, label, (x + 5, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Confidence score
        emotions = face_data['emotion']
        max_confidence = max(emotions.values())
        confidence_text = f"{max_confidence:.1f}%"
        cv2.putText(frame, confidence_text, (x, y + h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
    
    def draw_statistics(self, frame: np.ndarray) -> None:
        """Draw overall statistics"""
        height, width = frame.shape[:2]
        
        # Background for statistics
        cv2.rectangle(frame, (width - 200, 10), (width - 10, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 200, 10), (width - 10, 100), (100, 100, 100), 2)
        
        # Statistics text
        stats_text = [
            f"Frame: {self.frame_count}",
            f"Detections: {self.detection_count}",
            f"Active Faces: {len(self.face_tracker)}",
            f"Analysis Rate: {self.analysis_interval}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (width - 190, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def analyze_emotions(self, frame: np.ndarray) -> List[Dict]:
        """Analyze emotions with improved error handling and multiple face support"""
        try:
            # Use enforce_detection=False to handle cases with no clear faces
            results = DeepFace.analyze(frame, 
                                     actions=['emotion'], 
                                     enforce_detection=False,
                                     silent=True)
            
            # Handle both single face and multiple faces results
            if not isinstance(results, list):
                results = [results]
            
            processed_results = []
            for result in results:
                if 'emotion' in result and 'dominant_emotion' in result:
                    processed_results.append(result)
                    
                    # Update statistics
                    dominant = result['dominant_emotion']
                    self.emotion_stats[dominant]['count'] += 1
            
            self.detection_count += len(processed_results)
            return processed_results
            
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            return []
    
    def save_session_stats(self) -> None:
        """Save session statistics to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stats_file = f"emotion_stats_{timestamp}.json"
        
        # Convert defaultdict to regular dict for JSON serialization
        stats_data = {
            'session_info': {
                'total_frames': self.frame_count,
                'total_detections': self.detection_count,
                'analysis_interval': self.analysis_interval,
                'timestamp': timestamp
            },
            'emotion_statistics': dict(self.emotion_stats)
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
            print(f"Session statistics saved to: {stats_file}")
        except Exception as e:
            print(f"Error saving statistics: {e}")
    
    def run(self, camera_index: int = 0):
        """Main execution function"""
        print("Starting enhanced emotion recognition...")
        print("Controls:")
        print("  - 'q': Quit")
        print("  - 's': Save statistics")
        print("  - 'r': Reset statistics")
        print("  - '+': Increase analysis interval")
        print("  - '-': Decrease analysis interval")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Cannot access camera {camera_index}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            self.frame_count += 1
            
            # Analyze emotions at specified intervals
            if self.frame_count - self.last_analysis_frame >= self.analysis_interval:
                self.cached_results = self.analyze_emotions(frame)
                self.last_analysis_frame = self.frame_count
            
            # Track faces for stability
            if self.cached_results:
                tracked_faces = self.track_faces(self.cached_results)
                
                # Draw information for each face
                for i, face_data in enumerate(tracked_faces):
                    self.draw_face_info(frame, face_data, i)
                
                # Draw emotion bars for the first face (primary)
                if tracked_faces:
                    primary_face = tracked_faces[0]
                    self.draw_emotion_bars(frame, primary_face['emotion'])
            
            # Draw statistics
            self.draw_statistics(frame)
            
            # Instructions
            instructions = [
                "Controls: 'q'=quit, 's'=save, 'r'=reset, '+/-'=interval",
                f"Faces detected: {len(self.cached_results)}"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, frame.shape[0] - 30 + i * 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Display frame
            cv2.imshow("Enhanced Multi-Face Emotion Recognition", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("Saving session statistics...")
                self.save_session_stats()
            elif key == ord('r'):
                print("Resetting statistics...")
                self.emotion_stats.clear()
                self.detection_count = 0
                self.frame_count = 0
                self.face_tracker.clear()
                self.next_face_id = 0
            elif key == ord('+') or key == ord('='):
                self.analysis_interval = min(10, self.analysis_interval + 1)
                print(f"Analysis interval increased to: {self.analysis_interval}")
            elif key == ord('-'):
                self.analysis_interval = max(1, self.analysis_interval - 1)
                print(f"Analysis interval decreased to: {self.analysis_interval}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        if self.emotion_stats:
            print("\n=== FINAL SESSION STATISTICS ===")
            print(f"Total frames processed: {self.frame_count}")
            print(f"Total emotion detections: {self.detection_count}")
            print("Emotion distribution:")
            
            total_detections = sum(data['count'] for data in self.emotion_stats.values())
            for emotion, data in self.emotion_stats.items():
                percentage = (data['count'] / total_detections) * 100
                print(f"  {emotion}: {data['count']} ({percentage:.1f}%)")
        
        print("Enhanced emotion recognition stopped.")

def main():
    """Main function to run the emotion recognizer"""
    print("=== Enhanced Multi-Face Emotion Recognition ===")
    
    
    # Initialize and run recognizer
    recognizer = MultipleEmotionRecognizer()
    
    try:
        recognizer.run(0)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()