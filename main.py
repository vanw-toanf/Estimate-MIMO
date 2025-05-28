# ===============================
# MIMO Channel Estimation with AI
# ===============================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
n_tx = 4  # số anten truyền
n_rx = 4  # số anten nhận
n_samples_train = 10000  # số mẫu train
n_samples_test = 3000    # số mẫu test
snr_db_train = 20        # SNR cố định cho training
mod_order = 4            # QPSK: 4 symbols

# -------------------------------
# Generate dataset
# -------------------------------

def generate_dataset(n_samples, n_tx, n_rx, snr_db):
    X_data = []
    Y_data = []
    SNR_linear = 10**(snr_db/10)
    sigma2 = n_tx / SNR_linear

    for _ in range(n_samples):
        # Sinh ngẫu nhiên kênh H ~ Rayleigh
        H_real = np.random.randn(n_rx, n_tx) / np.sqrt(2)
        H_imag = np.random.randn(n_rx, n_tx) / np.sqrt(2)
        H = H_real + 1j * H_imag

        # Pilot matrix = Identity
        Xp = np.eye(n_tx)

        # Noise
        N_real = np.random.randn(n_rx, n_tx) * np.sqrt(sigma2/2)
        N_imag = np.random.randn(n_rx, n_tx) * np.sqrt(sigma2/2)
        N = N_real + 1j * N_imag

        # Tín hiệu nhận
        Yp = H @ Xp + N

        # Vector hóa
        X_vector = np.concatenate((Yp.real.flatten(), Yp.imag.flatten()))
        Y_vector = np.concatenate((H.real.flatten(), H.imag.flatten()))

        X_data.append(X_vector)
        Y_data.append(Y_vector)

    return np.array(X_data), np.array(Y_data)

# -------------------------------
# Define Neural Network
# -------------------------------

class ChannelEstimatorNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChannelEstimatorNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# Training
# -------------------------------

def train_model(model, X_train, Y_train, epochs=50, batch_size=64):
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.6f}")

# -------------------------------
# Channel Estimation (LS, MMSE)
# -------------------------------

def estimate_ls(Yp, Xp):
    return Yp @ np.linalg.inv(Xp)

def estimate_mmse(Yp, Xp, sigma2):
    Xt = Xp.conj().T
    inv = np.linalg.inv(Xp @ Xt + sigma2 * np.eye(Xp.shape[0]))
    return Yp @ Xt @ inv

# -------------------------------
# QPSK Modulation/Demodulation
# -------------------------------

def qpsk_mod(bits):
    symbols = (2*bits[:,0]-1) + 1j*(2*bits[:,1]-1)
    return symbols / np.sqrt(2)

def qpsk_demod(symbols):
    bits = np.zeros((len(symbols), 2))
    bits[:,0] = np.real(symbols) > 0
    bits[:,1] = np.imag(symbols) > 0
    return bits

# -------------------------------
# BER Calculation
# -------------------------------

def calculate_ber(H_estimator, model=None, snr_db_range=range(0, 31, 5)):
    ber_ls = []
    ber_mmse = []
    ber_ai = []

    for snr_db in snr_db_range:
        n_errors_ls = 0
        n_errors_mmse = 0
        n_errors_ai = 0
        n_total = 0

        SNR_linear = 10**(snr_db/10)
        sigma2 = n_tx / SNR_linear

        for _ in range(200):  # mỗi SNR lấy trung bình nhiều mẫu
            # Sinh H, pilot, noise
            H_real = np.random.randn(n_rx, n_tx) / np.sqrt(2)
            H_imag = np.random.randn(n_rx, n_tx) / np.sqrt(2)
            H = H_real + 1j * H_imag
            Xp = np.eye(n_tx)
            N_real = np.random.randn(n_rx, n_tx) * np.sqrt(sigma2/2)
            N_imag = np.random.randn(n_rx, n_tx) * np.sqrt(sigma2/2)
            N = N_real + 1j * N_imag
            Yp = H @ Xp + N

            # Tín hiệu dữ liệu random
            bits = np.random.randint(0, 2, (n_tx, 2))
            s = qpsk_mod(bits)
            s = s.reshape(-1,1)  # n_tx x 1
            noise_data_real = np.random.randn(n_rx, 1) * np.sqrt(sigma2/2)
            noise_data_imag = np.random.randn(n_rx, 1) * np.sqrt(sigma2/2)
            noise_data = noise_data_real + 1j * noise_data_imag
            y = H @ s + noise_data

            # --- LS
            H_ls = estimate_ls(Yp, Xp)
            s_hat_ls = np.linalg.pinv(H_ls) @ y
            bits_hat_ls = qpsk_demod(s_hat_ls.flatten())
            n_errors_ls += np.sum(bits != bits_hat_ls)

            # --- MMSE
            H_mmse = estimate_mmse(Yp, Xp, sigma2)
            s_hat_mmse = np.linalg.pinv(H_mmse) @ y
            bits_hat_mmse = qpsk_demod(s_hat_mmse.flatten())
            n_errors_mmse += np.sum(bits != bits_hat_mmse)

            # --- AI
            if model is not None:
                Yp_vec = np.concatenate((Yp.real.flatten(), Yp.imag.flatten()))
                Yp_vec_tensor = torch.FloatTensor(Yp_vec).unsqueeze(0)
                H_ai_vec = model(Yp_vec_tensor).detach().numpy().flatten()
                H_ai = H_ai_vec[:n_rx*n_tx].reshape(n_rx, n_tx) + 1j * H_ai_vec[n_rx*n_tx:].reshape(n_rx, n_tx)
                s_hat_ai = np.linalg.pinv(H_ai) @ y
                bits_hat_ai = qpsk_demod(s_hat_ai.flatten())
                n_errors_ai += np.sum(bits != bits_hat_ai)

            n_total += bits.size

        ber_ls.append(n_errors_ls / n_total)
        ber_mmse.append(n_errors_mmse / n_total)
        ber_ai.append(n_errors_ai / n_total)

    return ber_ls, ber_mmse, ber_ai

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    # Sinh dữ liệu
    print("Generating data...")
    X_train, Y_train = generate_dataset(n_samples_train, n_tx, n_rx, snr_db_train)

    # Khởi tạo model
    input_size = 2 * n_rx * n_tx
    output_size = 2 * n_rx * n_tx
    model = ChannelEstimatorNN(input_size, output_size)

    # Train model
    print("Training model...")
    train_model(model, X_train, Y_train, epochs=50)

    # Test và vẽ BER
    print("Evaluating BER...")
    snr_db_range = list(range(0, 31, 5))
    ber_ls, ber_mmse, ber_ai = calculate_ber(estimate_ls, model=model, snr_db_range=snr_db_range)

    # Vẽ kết quả
    plt.figure()
    plt.semilogy(snr_db_range, ber_ls, 'o-', label='LS')
    plt.semilogy(snr_db_range, ber_mmse, 's-', label='MMSE')
    plt.semilogy(snr_db_range, ber_ai, '^-', label='AI')
    plt.grid(True, which='both')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR')
    plt.legend()
    plt.show()
