import socket
import numpy as np
import time
import struct
import threading
import random
import pickle  # --- NEW: Import pickle ---

SENDER_HOST = '0.0.0.0'
SENDER_PORT = 65431
RECEIVER_HOST = 'receiver'
RECEIVER_PORT = 65432
VECTOR_SIZE = 512

class DynamicChannel:
    def __init__(self):
        self.current_noise = 0.05
        self.current_bandwidth = 10.0  # Mbps
        self.state_thread = threading.Thread(target=self.update_channel_state, daemon=True)
        self.state_thread.start()

    def update_channel_state(self):
        print("[Channel State] Dynamic state update thread started.")
        while True:
            noise_change = random.uniform(-0.01, 0.01)
            self.current_noise = np.clip(self.current_noise + noise_change, 0.0, 0.5)
            bw_change = random.uniform(-1.0, 1.0)
            self.current_bandwidth = np.clip(self.current_bandwidth + bw_change, 1.0, 20.0)
            time.sleep(3)

    def add_noise(self, vector_bytes):
        try:
            # --- NEW: The payload is now the vector itself, not bytes ---
            # (Note: sender.py sends the numpy array directly)
            if vector_bytes.shape[0] != VECTOR_SIZE:
                 print(f"  [Noise] Error: Expected vector size {VECTOR_SIZE}, got {vector_bytes.shape[0]}")
                 return vector_bytes
            
            noise = np.random.normal(0, self.current_noise, vector_bytes.shape)
            noisy_vector = vector_bytes + noise
            return noisy_vector
        except Exception as e:
            print(f"  [Noise] Error: {e}. Returning original vector.")
            return vector_bytes
            
    def recv_all(self, conn, n):
        buffer = b""
        while len(buffer) < n:
            chunk = conn.recv(n - len(buffer))
            if not chunk:
                return None
            buffer += chunk
        return buffer

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((SENDER_HOST, SENDER_PORT))
            s.listen()
            print("Channel is permanently listening for senders...")

            while True:
                print("\nWaiting for a new sender connection...")
                try:
                    conn, addr = s.accept()
                    with conn:
                        print(f"Sender connected from {addr}")
                        
                        while True:
                            header_bytes = self.recv_all(conn, 4)
                            if not header_bytes:
                                print(f"Sender {addr} disconnected (header read).")
                                break
                            
                            msg_len = struct.unpack('!I', header_bytes)[0]
                            # data is now the pickled_payload
                            data = self.recv_all(conn, msg_len)
                            if not data:
                                print(f"Sender {addr} disconnected (payload read).")
                                break
                            
                            # --- DYNAMIC LOGIC ---
                            noise_level = self.current_noise
                            bandwidth_mbps = self.current_bandwidth
                            
                            msg_len_megabits = (msg_len * 8) / 1_000_000
                            delay_sec = msg_len_megabits / bandwidth_mbps
                            
                            print(f"Received msg (Size: {msg_len} B). BW: {bandwidth_mbps:.2f} Mbps. Delaying for {delay_sec:.4f}s.")
                            time.sleep(delay_sec)
                            
                            # --- NEW: Unpickle the message ---
                            try:
                                message_dict = pickle.loads(data)
                                
                                if message_dict['type'] == "SEM":
                                    print(f"  Applying noise: {noise_level:.3f}")
                                    # Apply noise *to the vector array*
                                    noisy_vector = self.add_noise(message_dict['payload'])
                                    # Put the noisy vector back
                                    message_dict['payload'] = noisy_vector

                                elif message_dict['type'] == "RAW":
                                    print("  Forwarding RAW (no noise).")
                                    # No changes needed
                                
                                else:
                                    print(f"Unknown message type: {message_dict['type']}. Skipping.")
                                    continue
                                    
                            except Exception as e:
                                print(f"Error unpickling/processing message: {e}. Skipping.")
                                continue

                            # --- NEW: Re-pickle the (possibly modified) message ---
                            message_payload_bytes = pickle.dumps(message_dict)

                            # 3. Prepend network state and forward
                            noise_bytes = np.array([noise_level], dtype=np.float32).tobytes()
                            bw_bytes = np.array([bandwidth_mbps], dtype=np.float32).tobytes()
                            
                            # NEW FORMAT: noise | bandwidth | pickled_payload
                            message_to_forward = noise_bytes + b'|' + bw_bytes + b'|' + message_payload_bytes

                            try:
                                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rs:
                                    rs.connect((RECEIVER_HOST, RECEIVER_PORT))
                                    rs.sendall(message_to_forward)
                            except Exception as e:
                                print(f"Channel error forwarding to receiver: {e}")
                                
                except Exception as e:
                    print(f"Error in main connection loop: {e}. Resetting...")
                    time.sleep(1)

if __name__ == "__main__":
    channel = DynamicChannel()
    channel.run()