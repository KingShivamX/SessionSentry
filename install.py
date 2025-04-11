#!/usr/bin/env python
"""
Installation script for SessionSentry.
Sets up the environment and installs dependencies.
"""

import os
import sys
import subprocess
import shutil
import platform
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SessionSentry.Install")

def main():
    """Main installation function."""
    logger.info("Starting SessionSentry installation...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8 or higher is required.")
        sys.exit(1)
    
    # Check if running on Windows
    if platform.system() != "Windows":
        logger.error("SessionSentry is designed to run on Windows systems.")
        logger.error(f"Detected system: {platform.system()}")
        sys.exit(1)
    
    # Create virtual environment
    create_venv()
    
    # Install dependencies
    install_dependencies()
    
    # Create .env file if it doesn't exist
    create_env_file()
    
    # Initialize database
    initialize_database()
    
    # Create necessary directories
    create_directories()
    
    logger.info("SessionSentry installation completed successfully!")
    logger.info("To start the application, run: python src/main.py")
    logger.info("To start the dashboard, run: python src/dashboard/app.py")

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    if os.path.exists("venv"):
        logger.info("Virtual environment already exists, skipping creation.")
        return
    
    logger.info("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        logger.info("Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        sys.exit(1)

def install_dependencies():
    """Install dependencies from requirements.txt."""
    logger.info("Installing dependencies...")
    
    # Determine the pip executable
    if platform.system() == "Windows":
        pip = os.path.join("venv", "Scripts", "pip.exe")
    else:
        pip = os.path.join("venv", "bin", "pip")
    
    if not os.path.exists(pip):
        logger.error(f"Could not find pip at {pip}")
        sys.exit(1)
    
    try:
        subprocess.run([pip, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip, "install", "-r", "requirements.txt"], check=True)
        logger.info("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit(1)

def create_env_file():
    """Create .env file from .env.sample if it doesn't exist."""
    if os.path.exists(".env"):
        logger.info(".env file already exists, skipping creation.")
        return
    
    if os.path.exists(".env.sample"):
        logger.info("Creating .env file from .env.sample...")
        shutil.copy(".env.sample", ".env")
        logger.info(".env file created successfully.")
        logger.info("Please edit .env file with your configuration settings.")
    else:
        logger.warning(".env.sample file not found, creating empty .env file.")
        with open(".env", "w") as f:
            f.write("# SessionSentry configuration\n")
            f.write("DATABASE_URI=sqlite:///session_sentry.db\n")
            f.write("LOG_LEVEL=INFO\n")

def initialize_database():
    """Initialize the database schema."""
    logger.info("Initializing database...")
    
    # Determine the python executable
    if platform.system() == "Windows":
        python = os.path.join("venv", "Scripts", "python.exe")
    else:
        python = os.path.join("venv", "bin", "python")
    
    if not os.path.exists(python):
        logger.error(f"Could not find Python at {python}")
        sys.exit(1)
    
    # Create a temporary script to initialize the database
    with open("init_db.py", "w") as f:
        f.write("""
from src.data_collection.database import init_db

if __name__ == "__main__":
    init_db()
""")
    
    try:
        subprocess.run([python, "init_db.py"], check=True)
        logger.info("Database initialized successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary script
        if os.path.exists("init_db.py"):
            os.remove("init_db.py")

def create_directories():
    """Create necessary directories for the application."""
    directories = [
        "logs",
        "models",
        "templates",
        os.path.join("src", "dashboard", "static")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

if __name__ == "__main__":
    main() 