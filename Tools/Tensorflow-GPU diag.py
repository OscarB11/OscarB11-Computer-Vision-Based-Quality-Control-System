import tensorflow as tf
import wmi
import pycuda.driver as cuda
from pycuda.tools import DeviceData
from pycuda.gpuarray import GPUArray


print("TensorFlow version:", tf.__version__)

# Get the list of physical devices (GPUs)
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    print("Available GPU(s):")
    for gpu in physical_devices:
        print(gpu)
else:
    print("No GPUs available.")

        
    

cuda.init()
num_devices = cuda.Device.count()
print("Number of CUDA devices:", num_devices)

for i in range(num_devices):
    device = cuda.Device(i)
    print("\nDevice", i)
    print("  Name:", device.name())
    print("  Compute Capability:", device.compute_capability())
    print("  Total Memory:", round(device.total_memory() / (1024**3), 2), "GB")
    print("  Clock Rate:", device.get_attribute(cuda.device_attribute.CLOCK_RATE) / 1000, "MHz")
    

