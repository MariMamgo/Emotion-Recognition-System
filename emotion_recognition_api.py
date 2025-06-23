import cv2
import base64
import io
import json
import time
import threading
from PIL import Image
import google.generativeai as genai
from typing import Dict
import numpy as np
from datetime import datetime
import queue
import matplotlib.pyplot as plt
from collections import deque, Counter
import os

class LiveEmotionRecognizer:
    def __init__(self, api_key: str = "AIzaSyAIu8sBTvcURlDf1dAPs8sIo1CmXc2fnz0", model_name: str = "gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.emotions = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral", "contempt"]
        self.analysis_interval = 2.5  # Slightly faster analysis
        self.last_analysis_time = 0
        self.current_emotion_result = None
        self.is_analyzing = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.emotion_history = deque(maxlen=100)  # Use deque for better performance
        self.analysis_count = 0
        
        # Enhanced visualization
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Bright Green
            'sad': (255, 100, 100),    # Light Blue
            'angry': (0, 0, 255),      # Red
            'surprised': (0, 255, 255), # Yellow
            'fearful': (255, 0, 255),  # Magenta
            'disgusted': (0, 128, 128), # Dark Cyan
            'neutral': (128, 128, 128), # Gray
            'contempt': (128, 64, 0),  # Brown
            'error': (255, 255, 255)   # White
        }
        
        # Smoothing and trends
        self.emotion_smoothing = deque(maxlen=5)  # Last 5 emotions for smoothing
        self.confidence_history = deque(maxlen=20)  # Confidence trend
        
        # Visualization window
        self.show_chart = False
        self.chart_window_open = False

    def frame_to_base64(self, frame: np.ndarray) -> str:
        # Resize frame to reduce API costs while maintaining quality
        height, width = frame.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=80)  # Slightly lower quality for speed
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    def analyze_frame_emotion(self, frame: np.ndarray) -> Dict:
        start_time = time.time()
        try:
            img_base64 = self.frame_to_base64(frame)
            prompt = """
You are an expert facial emotion recognition system. Analyze the dominant emotion on the most prominent face in the given image.

Respond in ONLY valid JSON format:

{
  "faces_detected": <int>,
  "primary_emotion": "<emotion>",
  "confidence": <int from 0 to 100>,
  "emotions_detected": [<list of emotions>],
  "description": "<brief description>"
}

Valid emotions: happy, sad, angry, surprised, fearful, disgusted, neutral, contempt

Focus on the main face. Be confident in your assessment.
"""
            image_data = {'mime_type': 'image/jpeg', 'data': img_base64}
            response = self.model.generate_content([prompt, image_data])
            response_text = response.text.strip()
            
            # Better JSON extraction
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                result['timestamp'] = datetime.now().isoformat()
                result['analysis_success'] = True
                result['processing_time'] = time.time() - start_time
                
                # Add to smoothing
                emotion = result.get('primary_emotion', 'neutral')
                confidence = result.get('confidence', 0)
                self.emotion_smoothing.append(emotion)
                self.confidence_history.append(confidence)
                
            else:
                result = self._create_fallback_result(start_time)
        except json.JSONDecodeError:
            result = self._create_fallback_result(start_time, "JSON parsing failed")
        except Exception as e:
            result = self._create_error_result(start_time, str(e))
        return result

    def _create_fallback_result(self, start_time, reason="Could not parse response"):
        return {
            "faces_detected": 1,
            "primary_emotion": "neutral",
            "confidence": 50,
            "emotions_detected": ["neutral"],
            "description": f"Fallback: {reason}",
            "timestamp": datetime.now().isoformat(),
            "analysis_success": True,
            "processing_time": time.time() - start_time
        }

    def _create_error_result(self, start_time, error_msg):
        return {
            "faces_detected": 0,
            "primary_emotion": "error",
            "confidence": 0,
            "emotions_detected": [],
            "description": f"Analysis error: {error_msg}",
            "timestamp": datetime.now().isoformat(),
            "analysis_success": False,
            "processing_time": time.time() - start_time,
            "error": error_msg
        }

    def get_smoothed_emotion(self) -> str:
        """Get smoothed emotion using recent history"""
        if len(self.emotion_smoothing) < 3:
            return self.current_emotion_result.get('primary_emotion', 'neutral') if self.current_emotion_result else 'neutral'
        
        # Count emotions in recent history
        emotion_counts = Counter(self.emotion_smoothing)
        # Return most common emotion
        return emotion_counts.most_common(1)[0][0]

    def get_confidence_trend(self) -> float:
        """Get average confidence from recent analyses"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    def analysis_worker(self):
        while True:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break
                self.is_analyzing = True
                result = self.analyze_frame_emotion(frame)
                self.current_emotion_result = result
                self.emotion_history.append(result)
                self.analysis_count += 1
                self.is_analyzing = False
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis worker error: {e}")
                self.is_analyzing = False

    def draw_emotion_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]

        if self.current_emotion_result:
            result = self.current_emotion_result
            emotion = result.get('primary_emotion', 'Unknown')
            confidence = result.get('confidence', 0)
            faces_count = result.get('faces_detected', 0)
            processing_time = result.get('processing_time', 0)

            # Get smoothed emotion and trends
            smoothed_emotion = self.get_smoothed_emotion()
            avg_confidence = self.get_confidence_trend()

            color = self.emotion_colors.get(emotion.lower(), (255, 255, 255))
            
            # Main emotion panel with better design
            panel_width = 420
            panel_height = 180
            
            # Semi-transparent background
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), color, -1)
            cv2.addWeighted(overlay_frame, 0.8, overlay, 0.2, 0, overlay_frame)
            
            # Border
            cv2.rectangle(overlay_frame, (10, 10), (10 + panel_width, 10 + panel_height), color, 3)

            # Main emotion info
            cv2.putText(overlay_frame, f"EMOTION: {emotion.upper()}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Smoothed emotion if different
            if smoothed_emotion != emotion:
                cv2.putText(overlay_frame, f"Smoothed: {smoothed_emotion.upper()}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
                y_offset = 100
            else:
                y_offset = 70
            
            cv2.putText(overlay_frame, f"Confidence: {confidence}% (Avg: {avg_confidence:.1f}%)",
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(overlay_frame, f"Faces: {faces_count} | Time: {processing_time:.2f}s",
                        (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(overlay_frame, f"Analyses: {self.analysis_count}",
                        (20, y_offset + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Confidence bar
            bar_width = 300
            bar_height = 20
            bar_x = 20
            bar_y = y_offset + 80
            
            # Background bar
            cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Confidence fill
            fill_width = int((confidence / 100) * bar_width)
            cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
            # Border
            cv2.rectangle(overlay_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

        # Status indicator
        status_color = (0, 255, 0) if not self.is_analyzing else (0, 255, 255)
        status_text = "READY" if not self.is_analyzing else "ANALYZING..."
        cv2.putText(overlay_frame, status_text, (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Recent emotion trend (mini chart)
        if len(self.emotion_history) > 1:
            self.draw_mini_chart(overlay_frame, width, height)

        # Controls
        cv2.putText(overlay_frame, "Controls: q=quit | s=save | r=reset | c=chart | h=help",
                    (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame

    def draw_mini_chart(self, frame, width, height):
        """Draw a mini emotion trend chart on the frame"""
        if len(self.emotion_history) < 2:
            return
            
        chart_width = 200
        chart_height = 100
        chart_x = width - chart_width - 20
        chart_y = height - chart_height - 60
        
        # Background
        cv2.rectangle(frame, (chart_x, chart_y), (chart_x + chart_width, chart_y + chart_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (chart_x, chart_y), (chart_x + chart_width, chart_y + chart_height), (255, 255, 255), 2)
        
        # Get recent confidence values
        recent_confidences = [r.get('confidence', 0) for r in list(self.emotion_history)[-20:] if r.get('analysis_success', False)]
        
        if len(recent_confidences) > 1:
            # Draw confidence trend line
            step_x = chart_width / max(len(recent_confidences) - 1, 1)
            points = []
            
            for i, conf in enumerate(recent_confidences):
                x = chart_x + int(i * step_x)
                y = chart_y + chart_height - int((conf / 100) * chart_height)
                points.append((x, y))
            
            # Draw line
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 255), 2)
        
        # Chart label
        cv2.putText(frame, "Confidence Trend", (chart_x, chart_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def show_emotion_chart(self):
        """Show detailed emotion analysis chart"""
        if len(self.emotion_history) < 5:
            print("Not enough data for chart. Need at least 5 analyses.")
            return
            
        try:
            # Prepare data
            emotions = [r.get('primary_emotion', 'unknown') for r in self.emotion_history if r.get('analysis_success', False)]
            confidences = [r.get('confidence', 0) for r in self.emotion_history if r.get('analysis_success', False)]
            timestamps = [datetime.fromisoformat(r.get('timestamp', '')) for r in self.emotion_history if r.get('analysis_success', False)]
            
            if not emotions:
                print("No successful analyses to chart.")
                return
            
            # Create subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle('Emotion Recognition Analysis', fontsize=16, fontweight='bold')
            
            # 1. Emotion distribution (pie chart)
            emotion_counts = Counter(emotions)
            colors_list = [self.emotion_colors.get(emotion, (128, 128, 128)) for emotion in emotion_counts.keys()]
            colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors_list]
            
            ax1.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', 
                   colors=colors_normalized, startangle=90)
            ax1.set_title('Emotion Distribution')
            
            # 2. Confidence over time
            if timestamps:
                ax2.plot(timestamps, confidences, 'b-', linewidth=2, marker='o', markersize=4)
                ax2.set_title('Confidence Over Time')
                ax2.set_ylabel('Confidence (%)')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
            
            # 3. Recent emotion timeline
            recent_emotions = emotions[-20:]  # Last 20 emotions
            recent_times = timestamps[-20:] if timestamps else range(len(recent_emotions))
            
            # Map emotions to y-values for visualization
            emotion_y_map = {emotion: i for i, emotion in enumerate(set(recent_emotions))}
            y_values = [emotion_y_map[emotion] for emotion in recent_emotions]
            
            ax3.scatter(recent_times if timestamps else range(len(recent_emotions)), y_values, 
                       c=[emotion_counts[emotion] for emotion in recent_emotions], 
                       cmap='viridis', s=50, alpha=0.7)
            ax3.set_title('Recent Emotion Timeline')
            ax3.set_ylabel('Emotion Type')
            if timestamps:
                ax3.set_xlabel('Time')
            else:
                ax3.set_xlabel('Analysis Number')
            
            # Set y-tick labels to emotion names
            ax3.set_yticks(list(emotion_y_map.values()))
            ax3.set_yticklabels(list(emotion_y_map.keys()))
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show(block=False)  # Non-blocking show
            self.chart_window_open = True
            
        except Exception as e:
            print(f"Error creating chart: {e}")

    def print_help(self):
        """Print help information"""
        print("\n" + "="*50)
        print("LIVE EMOTION RECOGNITION - CONTROLS")
        print("="*50)
        print("q - Quit application")
        print("s - Save session results to JSON file")
        print("r - Reset emotion history and counters")
        print("c - Show detailed emotion analysis chart")
        print("h - Show this help menu")
        print("="*50 + "\n")

    def save_session_results(self):
        if not self.emotion_history:
            print("No analysis results to save.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_session_{timestamp}.json"
        
        # Enhanced session data
        session_data = {
            "session_info": {
                "start_time": self.emotion_history[0]['timestamp'] if self.emotion_history else None,
                "end_time": self.emotion_history[-1]['timestamp'] if self.emotion_history else None,
                "total_analyses": len(self.emotion_history),
                "successful_analyses": len([r for r in self.emotion_history if r.get('analysis_success', False)]),
                "analysis_interval": self.analysis_interval,
                "average_confidence": self.get_confidence_trend(),
                "dominant_emotion": self.get_smoothed_emotion()
            },
            "emotion_history": list(self.emotion_history),
            "emotion_summary": self.get_emotion_summary(),
            "performance_stats": {
                "average_processing_time": np.mean([r.get('processing_time', 0) for r in self.emotion_history if r.get('processing_time')]),
                "total_processing_time": sum([r.get('processing_time', 0) for r in self.emotion_history if r.get('processing_time')])
            }
        }
        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            print(f"Session results saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def get_emotion_summary(self) -> Dict:
        if not self.emotion_history:
            return {}
        
        successful_results = [r for r in self.emotion_history if r.get('analysis_success', False)]
        if not successful_results:
            return {}
            
        emotion_counts = Counter(r.get('primary_emotion', 'unknown') for r in successful_results)
        total = len(successful_results)
        
        emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
        
        return {
            "total_analyses": len(self.emotion_history),
            "successful_analyses": len(successful_results),
            "emotion_counts": dict(emotion_counts),
            "emotion_percentages": emotion_percentages,
            "dominant_emotion": max(emotion_counts, key=emotion_counts.get) if emotion_counts else "none",
            "average_confidence": self.get_confidence_trend()
        }

    def start_live_recognition(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera opened successfully!")
        self.print_help()
        
        analysis_thread = threading.Thread(target=self.analysis_worker, daemon=True)
        analysis_thread.start()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                current_time = time.time()
                if (current_time - self.last_analysis_time) >= self.analysis_interval:
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                        self.last_analysis_time = current_time
                    except queue.Full:
                        pass
                
                display_frame = self.draw_emotion_overlay(frame)
                cv2.imshow('Live Emotion Recognition - Enhanced', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_session_results()
                elif key == ord('r'):
                    self.emotion_history.clear()
                    self.emotion_smoothing.clear()
                    self.confidence_history.clear()
                    self.analysis_count = 0
                    self.current_emotion_result = None
                    print("Session reset!")
                elif key == ord('c'):
                    self.show_emotion_chart()
                elif key == ord('h'):
                    self.print_help()
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.frame_queue.put(None)
            cap.release()
            cv2.destroyAllWindows()
            if plt.get_fignums():  # Close any open matplotlib windows
                plt.close('all')
            
            if self.emotion_history:
                summary = self.get_emotion_summary()
                print(f"\n=== FINAL SESSION SUMMARY ===")
                print(f"Total analyses: {summary['total_analyses']}")
                print(f"Successful analyses: {summary['successful_analyses']}")
                print(f"Average confidence: {summary['average_confidence']:.1f}%")
                if summary.get('dominant_emotion'):
                    print(f"Dominant emotion: {summary['dominant_emotion']}")
                    print("Emotion distribution:")
                    for emotion, percentage in summary['emotion_percentages'].items():
                        print(f"  {emotion}: {percentage:.1f}%")
                print("=" * 30)

# Run the application
if __name__ == "__main__":
    print("=== Enhanced Live Emotion Recognition System ===\n")
    recognizer = LiveEmotionRecognizer()
    recognizer.start_live_recognition(0)