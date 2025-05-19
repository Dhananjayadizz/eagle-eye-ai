import os

def create_directory_structure():
    # Define the directory structure
    directories = [
        'app',
        'app/models',
        'app/utils',
        'app/core',
        'app/static',
        'app/templates',
        'app/config',
        'app/data',
        'app/static/css',
        'app/static/js',
        'app/static/images',
        'app/data/uploads',
        'app/data/exports'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directory_structure() 