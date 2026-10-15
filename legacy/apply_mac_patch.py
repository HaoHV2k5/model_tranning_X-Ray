"""
Chạy 1 lần TRONG thư mục MedicalPatchNet (nơi có trainClassification.py):
    python3 apply_mac_patch.py
Script sửa code gốc (viết cho GPU NVIDIA) để chạy trên Mac M1 (MPS) với subset CheXpert của bạn.
File gốc được backup thành *.orig
"""
import shutil, sys, pathlib

def patch(fname, pairs):
    p = pathlib.Path(fname)
    if not p.exists():
        sys.exit(f"Không thấy {fname} - hãy chạy script trong thư mục MedicalPatchNet")
    src = p.read_text()
    if not pathlib.Path(fname + ".orig").exists():
        shutil.copy(p, fname + ".orig")
    for old, new in pairs:
        if old not in src:
            sys.exit(f"[{fname}] không tìm thấy đoạn cần sửa:\n{old}\n(repo có thể đã đổi - báo lại để sửa tay)")
        src = src.replace(old, new)
    p.write_text(src)
    print("patched", fname)

# ---------- ChexpertDataset.py ----------
patch("ChexpertDataset.py", [
    ('import torch\n\nTEST = "test"',
     'import torch\nimport os\n\nTEST = "test"'),
    ('''if ARG.ON_HPC:
    BASE_PATH = "/hpcwork/p0021834/workspace_patrick/datasets/CheXpert-v1.0_1024/"
else:
    BASE_PATH = "/home/patrick/datasets/CheXpert-v1.0_1024/"''',
     '''# Thư mục chứa train.csv / val.csv / test.csv / images/ (đặt bằng biến môi trường CHEXPERT_ROOT)
BASE_PATH = os.environ.get("CHEXPERT_ROOT", "./chexpert_subset")
if not BASE_PATH.endswith("/"): BASE_PATH += "/"'''),
    ('''    TRAIN:"train_visualCheXbert.csv",
    #TRAIN:"train.csv", #for the classic training data
    VALIDATE:"valid.csv",
    TEST:"test_labels.csv",''',
     '''    TRAIN:"train.csv",
    VALIDATE:"val.csv",
    TEST:"test.csv",'''),
    ('''    VALIDATE:5,
    TEST:1,''',
     '''    VALIDATE:5,
    TEST:5,'''),
])

# ---------- trainClassification.py ----------
patch("trainClassification.py", [
    ("import wandb\n",
     "try:\n    import wandb\nexcept ImportError:\n    wandb = None\n"),
    ("NUM_PROC = 24 #number of processes used to load data",
     "NUM_PROC = 4 #number of processes used to load data (Mac: 4-6 là đủ)"),
    ('''        with autocast(device_type='cuda'):
            output = self.model(img)
            loss = self.criterion(output,label)
        
        self.ampScaler.scale(loss).backward()
        self.ampScaler.step(self.optimizer)
        self.ampScaler.update()
''',
     '''        output = self.model(img)
        loss = self.criterion(output,label)
        loss.backward()
        self.optimizer.step()
'''),
    ("        pin_memory=True,", "        pin_memory=False,"),
    ("        self.ampScaler = GradScaler()\n", "        self.ampScaler = None\n"),
    ("    model = ModelClass(patchSize=ARG.PATCH_SIZE,outFeatures=14)\n    #compile model for speedup\n    model = torch.compile(model)\n",
     "    model = ModelClass(patchSize=ARG.PATCH_SIZE,outFeatures=14)\n"),
    ("    model.load_state_dict(loadedWeights)\n    model = torch.compile(model)\n",
     "    model.load_state_dict(loadedWeights)\n"),
    ("bootstrapNum = 100000", "bootstrapNum = 1000"),
    ("    device = 0  # shorthand for cuda:0\n",
     "    device = 'mps' if torch.backends.mps.is_available() else 'cpu'\n    print('Using device:', device)\n"),
])

print("\nXong. Backup: ChexpertDataset.py.orig, trainClassification.py.orig")
