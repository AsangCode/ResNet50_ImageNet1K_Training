#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${YELLOW}[STATUS]${NC} $1"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to print error messages and exit
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if the EBS volume exists
check_ebs() {
    print_status "Checking EBS volume..."
    if [ ! -e /dev/nvme2n1 ]; then
        print_error "EBS volume /dev/nvme2n1 not found!"
    fi
    print_success "EBS volume found"
}

# Create and mount directory
setup_mount() {
    print_status "Creating mount point..."
    sudo mkdir -p /mnt/imagenet || print_error "Failed to create mount directory"
    
    # Check if already mounted
    if mountpoint -q /mnt/imagenet; then
        print_success "EBS volume already mounted at /mnt/imagenet"
        return 0
    fi
    
    print_status "Checking filesystem..."
    sudo file -s /dev/nvme2n1
    
    print_status "Mounting EBS volume..."
    sudo mount /dev/nvme2n1 /mnt/imagenet || print_error "Failed to mount EBS volume"
    print_success "EBS volume mounted successfully"
}

# Setup Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Install Python 3.12 venv if not present
    if ! command -v python3.12 &> /dev/null; then
        print_status "Installing Python 3.12..."
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt update
        sudo apt install python3.12 python3.12-venv -y
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "/mnt/imagenet/resnet50" ]; then
        print_status "Creating virtual environment..."
        python3.12 -m venv /mnt/imagenet/resnet50 || print_error "Failed to create virtual environment"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source /mnt/imagenet/resnet50/bin/activate || print_error "Failed to activate virtual environment"
    
    # Install/upgrade pip
    print_status "Upgrading pip..."
    python -m pip install --upgrade pip || print_error "Failed to upgrade pip"
    
    print_success "Python environment setup complete"
}

# Function to handle repository setup choices
handle_repo_choice() {
    echo -e "\n${YELLOW}Repository Options:${NC}"
    echo "1) Reset all local changes and pull latest"
    echo "2) Backup local changes and pull latest"
    echo "3) Try to merge local changes with latest"
    echo "4) Keep local changes only (skip pull)"
    echo "5) Exit script"
    
    read -p "Choose an option (1-5): " choice
    
    case $choice in
        1)
            print_status "Resetting local changes..."
            git reset --hard HEAD
            git pull origin main
            ;;
        2)
            print_status "Backing up local changes..."
            timestamp=$(date +%Y%m%d_%H%M%S)
            backup_dir="/mnt/imagenet/code_backups_${timestamp}"
            mkdir -p "$backup_dir"
            cp -r * "$backup_dir/"
            print_success "Backup created at: $backup_dir"
            
            print_status "Resetting and pulling latest changes..."
            git reset --hard HEAD
            git pull origin main
            ;;
        3)
            print_status "Stashing local changes..."
            git stash
            git pull origin main
            print_status "Reapplying local changes..."
            git stash pop
            ;;
        4)
            print_status "Keeping local changes, skipping pull..."
            ;;
        5)
            print_status "Exiting script..."
            exit 0
            ;;
        *)
            print_error "Invalid choice!"
            ;;
    esac
}

# Clone and setup repository
setup_repo() {
    print_status "Setting up repository..."
    
    # Navigate to imagenet directory
    cd /mnt/imagenet || print_error "Failed to change directory"
    
    # Create training directory
    mkdir -p ResNet50_ImageNet1K_Training
    cd ResNet50_ImageNet1K_Training || print_error "Failed to create/enter training directory"
    
    # Handle repository setup
    if [ ! -d ".git" ]; then
        print_status "Cloning repository..."
        git clone https://github.com/AsangCode/ResNet50_ImageNet1K_Training.git . || print_error "Failed to clone repository"
    else
        print_status "Repository already exists..."
        
        # Check for local changes
        if [ -n "$(git status --porcelain)" ]; then
            print_status "Local changes detected!"
            handle_repo_choice
        else
            print_status "No local changes, pulling latest..."
            git pull origin main
        fi
    fi
    
    # Install requirements
    print_status "Installing requirements..."
    pip install -r requirements.txt || print_error "Failed to install requirements"
    
    print_success "Repository setup complete"
}

# Start training
start_training() {
    print_status "Starting training..."
    python3 train.py
}

# Main execution
main() {
    print_status "Starting setup pipeline..."
    
    # Execute all steps
    check_ebs
    setup_mount
    setup_python
    setup_repo
    start_training
    
    print_success "Setup complete!"
}

# Execute main function
main
