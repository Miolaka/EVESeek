import torch
import time

# =========================================
#  ENVIRONMENT INFO
# =========================================
print("=" * 50)
print("🔍 PyTorch Environment Check")
print("=" * 50)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA runtime version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device index: {torch.cuda.current_device()}")
else:
    print("⚠️  No GPU detected. Training will run on CPU only.")

# =========================================
#  DEFINE A SIMPLE MODEL
# =========================================
print("\nBuilding test model...")
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

# Dummy dataset
x = torch.randn(5000, 1000)
y = torch.randn(5000, 10)

# =========================================
#  TRAIN ON CPU
# =========================================
print("\nRunning on CPU...")
device = torch.device("cpu")
model_cpu = model.to(device)
optimizer = torch.optim.Adam(model_cpu.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

start = time.time()
for epoch in range(50):
    optimizer.zero_grad()
    loss = criterion(model_cpu(x), y)
    loss.backward()
    optimizer.step()
cpu_time = time.time() - start
print(f"✅ CPU training finished in {cpu_time:.2f} sec")

# =========================================
#  TRAIN ON GPU (if available)
# =========================================
if torch.cuda.is_available():
    print("\nRunning on GPU...")
    device = torch.device("cuda")
    model_gpu = model.to(device)
    x_gpu = x.to(device)
    y_gpu = y.to(device)
    optimizer = torch.optim.Adam(model_gpu.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    torch.cuda.synchronize()
    start = time.time()
    for epoch in range(50):
        optimizer.zero_grad()
        loss = criterion(model_gpu(x_gpu), y_gpu)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    print(f"✅ GPU training finished in {gpu_time:.2f} sec")
    print(f"🚀 Speedup: {cpu_time / gpu_time:.1f}× faster on GPU!")
else:
    print("\n⚠️  CUDA not available, skipping GPU test.")

# =========================================
#  OPTIONAL: GPU MEMORY STATS
# =========================================
if torch.cuda.is_available():
    print("\n📊 GPU Memory Info:")
    mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"  Memory allocated: {mem_alloc:.1f} MB")
    print(f"  Memory reserved:  {mem_reserved:.1f} MB")

print("\n✅ All tests completed successfully.")
print("=" * 50)
torch.save(model_gpu.state_dict(), "test_model.pt")

