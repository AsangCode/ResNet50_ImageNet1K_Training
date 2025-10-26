# ImageNet-1K Download on EC2 with EBS

This guide explains how to **provision an EBS volume**, attach it to an EC2 instance, and download the **ImageNet-1K dataset from Hugging Face** directly to the volume. It also includes the **non-interactive download method** and resolves common issues.

---

## 1️⃣ Provision and Attach EBS Volume

1. Go to AWS EC2 → Volumes → Create Volume.

   * **Size:** 420 GB
   * **Type:** gp3 (or gp2)
   * **Availability Zone:** Same as your EC2 instance

2. Attach the volume to your EC2 instance (note the device name, e.g., `/dev/sdf`).

3. Connect to your EC2 instance:

```bash
ssh -i "your-key.pem" ec2-user@<EC2_PUBLIC_IP>
```

4. Verify attached disks:

```bash
lsblk
```

5. Format the new volume (replace `/dev/nvme4n1` with your device if different):

```bash
sudo mkfs -t ext4 /dev/nvme4n1
```

6. Mount the volume:

```bash
sudo mkdir /mnt/imagenet
sudo mount /dev/nvme4n1 /mnt/imagenet
```

7. Set ownership so the current user (`root`) can write:

```bash
sudo chown -R root:root /mnt/imagenet
```

> ⚠️ If you create another user, replace `root` with that username.

8. Verify mount and free space:

```bash
df -h /mnt/imagenet
```

---

## 2️⃣ Install Required Libraries

```bash
pip3 install --upgrade datasets tqdm huggingface_hub pillow
```

* **`datasets`**: Hugging Face datasets library
* **`tqdm`**: Progress bar
* **`huggingface_hub`**: Authentication & download
* **`pillow`**: Required to decode and save images

---

## 3️⃣ Authenticate Hugging Face

```bash
hf auth login
```

Follow the prompt and enter your Hugging Face token.

---

## 4️⃣ Download ImageNet-1K Dataset

### Python Script: `download_imagenet.py`

```python
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import os

# Dataset configuration
dataset_name = "ILSVRC/imagenet-1k"
save_root = Path("/mnt/imagenet")
os.makedirs(save_root, exist_ok=True)

def save_split(split_name):
    print(f"\n📦 Downloading {split_name} split...")
    ds = load_dataset(dataset_name, split=split_name, cache_dir=str(save_root))
    split_dir = save_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(tqdm(ds, desc=f"Saving {split_name} images")):
        label = item["label"]
        label_dir = split_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)

        file_path = label_dir / f"{idx}.JPEG"
        # Skip if file already exists
        if file_path.exists():
            continue

        image = item["image"]
        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(file_path)

    print(f"✅ Finished {split_name} → {split_dir}")

if __name__ == "__main__":
    save_split("train")
    save_split("validation")
    print("\n🎉 All ImageNet images saved under /mnt/imagenet/")
```

### Notes:

* The `id` field is no longer used; we use `idx` to ensure unique filenames.
* All files and temporary Hugging Face caches are stored on `/mnt/imagenet`.

---

## 5️⃣ Run Script in Non-Interactive Mode

To download safely without stopping if the SSH session disconnects:

```bash
nohup python3 /mnt/imagenet/download_imagenet.py > /mnt/imagenet/download.log 2>&1 &
```

* `nohup` → keeps the process running after logout
* `> /mnt/imagenet/download.log 2>&1` → saves logs
* `&` → runs in background

Check progress anytime:

```bash
tail -f /mnt/imagenet/download.log
```

> Press `Ctrl+C` to exit `tail`; the download **will continue running in the background**.

---

## 6️⃣ Folder Structure After Download

```
/mnt/imagenet/
├── train/
│   ├── 0/
│   │   ├── 0.JPEG
│   │   ├── 1.JPEG
│   │   └── ...
│   ├── 1/
│   └── ...
├── validation/
│   ├── 0/
│   └── ...
└── download.log
```

---

## 7️⃣ Summary of Fixes in This Guide

* **Cache directory** set to `/mnt/imagenet` to prevent root volume overflow
* **Pillow installed** to decode and save images
* **`id` replaced with `idx`** for unique filenames
* **Non-interactive download** with `nohup` so it continues after disconnect
* Instructions generalized for **`root` user** or any custom user

---

You are now ready to use `/mnt/imagenet` for training models directly on the ImageNet-1K dataset.
