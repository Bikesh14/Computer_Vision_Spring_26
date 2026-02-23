import cv2
import mediapipe as mp
import time
import os
import subprocess
import matplotlib.pyplot as plt

# Configuration
PDF_PATH = 'pdf_to_read.pdf'
MOVIE_PATH = 'movie_to_watch.mp4'

class NativeResearchApp:
    def __init__(self):
        self.stats = {} # Stores results for final comparison
        
        # Setup MediaPipe
        base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
        self.options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1
        )

    def run_trial(self, trial_name, media_path):
        print(f"\n>>> PHASE: {trial_name}")
        subprocess.Popen(['open', media_path])
        
        blink_count = 0
        eye_closed = False
        start_time = time.time()
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        window_name = f"Monitor: {trial_name}"
        # WINDOW_AUTOSIZE so it snaps to the image dimensions
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) 

        with mp.tasks.vision.FaceLandmarker.create_from_options(self.options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                display_w, display_h = 300, 200 
                frame = cv2.resize(frame, (display_w, display_h))
                frame = cv2.flip(frame, 1)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                current_elapsed = time.time() - start_time
                results = landmarker.detect_for_video(mp_image, int(current_elapsed * 1000))

                if results.face_blendshapes:
                    scores = results.face_blendshapes[0]
                    l = next(i.score for i in scores if i.category_name == 'eyeBlinkLeft')
                    r = next(i.score for i in scores if i.category_name == 'eyeBlinkRight')
                    conf = (l + r) / 2.0

                    if conf > 0.45:
                        if not eye_closed:
                            blink_count += 1
                            eye_closed = True
                    else:
                        eye_closed = False

                # UI on the tiny frame
                cv2.putText(frame, f"No. of blinks: {blink_count}", (5, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(frame, "Press 'D' if Done", (5, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.imshow(window_name, frame)
                
                # Update position every frame to ensure it stays pinned
                cv2.moveWindow(window_name, 1100, 50) 
                
                if cv2.waitKey(1) & 0xFF == ord('d'): break

        total_duration = time.time() - start_time
        self.stats[trial_name] = {
            "time": total_duration,
            "blinks": blink_count,
            "rate": (blink_count / total_duration) * 60 if total_duration > 0 else 0
        }
        cap.release()
        cv2.destroyWindow(window_name)

    def display_final_report(self):
        """Generates a comparison bar chart and printed summary."""
        print("\n" + "="*40)
        print(f"{'METRIC':<20} | {'READING PDF':<12} | {'WATCHING MOVIE':<12}")
        print("-" * 50)
        
        pdf = self.stats.get("Reading_PDF", {"time":0, "blinks":0, "rate":0})
        mov = self.stats.get("Watching_Movie", {"time":0, "blinks":0, "rate":0})

        print(f"{'Total Time (s)':<20} | {pdf['time']:<12.1f} | {mov['time']:<12.1f}")
        print(f"{'Total Blinks':<20} | {pdf['blinks']:<12} | {mov['blinks']:<12}")
        print(f"{'Blink Rate (BPM)':<20} | {pdf['rate']:<12.2f} | {mov['rate']:<12.2f}")
        print("="*40)

        # Comparison Plot
        labels = ['Reading PDF', 'Watching Movie']
        rates = [pdf['rate'], mov['rate']]
        counts = [pdf['blinks'], mov['blinks']]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Rate Plot
        ax1.bar(labels, rates, color=['blue', 'red'], alpha=0.7)
        ax1.set_title('Blink Rate (Blinks per Minute)')
        ax1.set_ylabel('BPM')

        # Total Count Plot
        ax2.bar(labels, counts, color=['blue', 'red'], alpha=0.7)
        ax2.set_title('Total Blink Count')
        ax2.set_ylabel('Blinks')

        plt.tight_layout()
        filename = f"blink_comparison.png"
        plt.savefig(filename)
        plt.show()

def main():
    app = NativeResearchApp()
    
    if not os.path.exists(PDF_PATH) or not os.path.exists(MOVIE_PATH):
        print("Missing files! Please check file paths.")
        return

    # Phase 1: PDF
    app.run_trial("Reading_PDF", PDF_PATH)
    
    print("\nNext phase in 3 seconds...")
    time.sleep(3)
    
    # Phase 2: Movie
    app.run_trial("Watching_Movie", MOVIE_PATH)
    
    # Report
    app.display_final_report()

if __name__ == "__main__":
    main()