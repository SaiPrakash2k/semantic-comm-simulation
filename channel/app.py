import socket
import numpy as np

SENDER_HOST = '0.0.0.0'
SENDER_PORT = 65431
RECEIVER_HOST = 'receiver'
RECEIVER_PORT = 65432

# --- NOISE CONFIGURATION ---
# The intensity of the noise. Higher value = more corruption.
NOISE_LEVEL = 0.1 
# The size of the feature vector from ResNet18 is 512
VECTOR_SIZE = 512 

def add_noise(vector_bytes):
    """Deserializes the vector, adds Gaussian noise, and re-serializes."""
    original_vector = np.frombuffer(vector_bytes, dtype=np.float32)
    noise = np.random.normal(0, NOISE_LEVEL, original_vector.shape)
    noisy_vector = original_vector
    return noisy_vector.tobytes()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((SENDER_HOST, SENDER_PORT))
    s.listen()
    print("Channel is waiting for sender...")
    conn, addr = s.accept()
    with conn:
        print(f"Sender connected from {addr}")
        while True:
            data = conn.recv(4096) # Increased buffer for vector
            if not data:
                break
            
            # Separate the label from the vector bytes
            label_bytes, vector_bytes = data.split(b'|', 1)
            
            # Corrupt the vector
            noisy_vector_bytes = add_noise(vector_bytes)
            
            print(f"Received semantics for '{label_bytes.decode()}'. Adding noise and forwarding.")
            
            message_to_forward = label_bytes + b'|' + noisy_vector_bytes

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receiver_socket:
                receiver_socket.connect((RECEIVER_HOST, RECEIVER_PORT))
                receiver_socket.sendall(message_to_forward)