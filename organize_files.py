import os
import shutil

def organize_files():
    # Define file mappings (source -> destination)
    file_mappings = {
        # Models
        'yolov8n.pt': 'app/models/',
        'frontier_anomaly_model.pkl': 'app/models/',
        'scaler.pkl': 'app/models/',
        'frontier_classifier.pkl': 'app/models/',
        
        # Core functionality
        'vehicle_tracker.py': 'app/core/',
        'motion_detection.py': 'app/core/',
        'motion_analysis.py': 'app/core/',
        'yolo_tracking.py': 'app/core/',
        'trajectory_prediction.py': 'app/core/',
        'sort.py': 'app/core/',
        
        # Utils
        'gps_module.py': 'app/utils/',
        'model_convert.py': 'app/utils/',
        
        # Training scripts
        'train_anomaly_model.py': 'app/models/',
        'frontier_classifier_test.py': 'app/models/',
        
        # Main application
        'app.py': 'app/',
        'blockchain.py': 'app/',
        'run.py': '.',
        
        # Static files
        'test_image.jpg': 'app/static/images/',
        
        # Templates (if any)
        'templates/*': 'app/templates/',
        
        # Data
        'exports/*': 'app/data/exports/'
    }
    
    # Move files to their new locations
    for source, dest in file_mappings.items():
        if '*' in source:
            # Handle wildcard patterns
            pattern = source.replace('*', '')
            for file in os.listdir('.'):
                if file.startswith(pattern) or file.endswith(pattern):
                    try:
                        shutil.move(file, os.path.join(dest, file))
                        print(f"Moved {file} to {dest}")
                    except Exception as e:
                        print(f"Error moving {file}: {e}")
        else:
            # Handle individual files
            if os.path.exists(source):
                try:
                    shutil.move(source, os.path.join(dest, source))
                    print(f"Moved {source} to {dest}")
                except Exception as e:
                    print(f"Error moving {source}: {e}")

if __name__ == "__main__":
    organize_files() 