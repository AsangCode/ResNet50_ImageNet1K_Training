# ImageNet-1K Dataset Setup Guide

This guide explains how to download, extract, and organize the ImageNet-1K dataset using AWS EC2 and EBS.

## 1. EC2 and EBS Setup

### EC2 Instance Requirements
- Instance Type: c5.4xlarge (recommended)
  - 16 vCPUs
  - 32GB RAM
  - Good network performance
- AMI: Ubuntu Server 22.04 LTS

### EBS Volume Setup
1. Create EBS Volume:
   - Size: 500GB
   - Type: gp3
   - Same availability zone as EC2

2. Attach to EC2:
   - Note device name (e.g., /dev/sdf)
   - Will appear as /dev/nvme1n1 in instance

3. Format and Mount:
```bash
# Check device name
lsblk

# Format EBS
sudo mkfs -t ext4 /dev/nvme1n1

# Create mount point
sudo mkdir /mnt/imagenet

# Mount
sudo mount /dev/nvme1n1 /mnt/imagenet

# Set permissions
sudo chown -R ubuntu:ubuntu /mnt/imagenet
```

## 2. Download Dataset

### Training Data
```bash
# Create directory structure
mkdir -p /mnt/imagenet/raw
cd /mnt/imagenet/raw

# Download using aria2
aria2c "https://academictorrents.com/download/a306397ccf9c2ead27155983c254227c0fd938e2.torrent" \
    --seed-time=0 \
    --max-concurrent-downloads=5 \
    --split=5 \
    --min-split-size=1M
```

### Validation Data
```bash
# Download validation data (requires ImageNet account)
aria2c "https://academictorrents.com/download/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent" \
    --seed-time=0 \
    --max-concurrent-downloads=5 \
    --split=5 \
    --min-split-size=1M
```

## 3. Extract and Organize

### Training Data
```bash
# Create extraction directory
mkdir -p /mnt/imagenet/extracted/train
cd /mnt/imagenet/extracted/train

# Extract main archive
tar xf /mnt/imagenet/raw/ILSVRC2012_img_train.tar

# Extract individual class archives with progress
total=$(ls *.tar | wc -l)
current=0
for f in *.tar; do
    current=$((current + 1))
    echo "Processing $current/$total: $f"
    d=`basename $f .tar`
    mkdir -p $d
    cd $d
    tar xf ../$f
    cd ..
    rm $f
    echo "Progress: $((current * 100 / total))% complete"
    echo "----------------------------------------"
done
```

### Validation Data
```bash
# Create and move to validation directory
mkdir -p /mnt/imagenet/extracted/val
cd /mnt/imagenet/extracted/val

# Extract validation archive
tar xf /mnt/imagenet/raw/ILSVRC2012_img_val.tar

# Download and run validation preparation script
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x valprep.sh
./valprep.sh
```

## 4. Final Directory Structure
```
/mnt/imagenet/
├── raw/                    # Raw downloaded files
│   ├── ILSVRC2012_img_train.tar
│   └── ILSVRC2012_img_val.tar
├── extracted/
│   ├── train/             # Training images
│   │   ├── n01440764/    # Class folder
│   │   │   ├── image1.JPEG
│   │   │   └── ...
│   │   └── ...
│   └── val/              # Validation images
│       ├── n01440764/
│       └── ...
```

## 5. Preservation Steps

After setup:
1. Stop (don't terminate) EC2 instance
2. Note EBS volume ID from AWS Console
3. You can terminate instance but keep EBS volume
4. EBS volume can be attached to any instance for training

## 6. Expected Times
- Download: 2-4 hours
- Training Extraction: 1-2 hours
- Validation Setup: 30 minutes
- Total: 4-7 hours

## 7. Cost Estimates
- EC2 (c5.4xlarge): ~$0.68/hour
- EBS (500GB gp3): ~$40/month
- Setup Cost: ~$5-7
- Monthly Storage: ~$40

## 8. Monitoring Commands
```bash
# Check extraction progress
ls -d */ | wc -l  # Count directories
ls *.tar | wc -l  # Count remaining archives

# Check disk usage
df -h /mnt/imagenet
```

## Notes
- Keep EBS volume ID safe
- Don't delete EBS volume unless you're sure
- Dataset size: ~150GB each for train and val
- Total classes: 1000
- Total images: ~1.2 million training, 50,000 validation
