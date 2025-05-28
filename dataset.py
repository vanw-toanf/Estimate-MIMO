import torch
from torch.utils.data import Dataset
import numpy as np

class MIMODataset(Dataset):
    # def __init__(self, num_samples=1000, snr_db=10, Nt=2, Nr=2):
    def __init__(self, num_samples=1000, snr_db_range=(0, 20), Nt=2, Nr=2):
        self.snr_db_range = snr_db_range
        self.num_samples = num_samples
        # self.snr_db = snr_db
        self.Nt = Nt  # số anten phát
        self.Nr = Nr  # số anten thu
        self.pilot_length = Nt  # Đơn giản lấy pilot_length = Nt (ma trận đơn vị)

        self.x, self.y = self.generate_data()

    def generate_data(self):
        x = []
        y = []

        random_snr_db = np.random.uniform(self.snr_db_range[0], self.snr_db_range[1])
        snr_linear = 10 ** (random_snr_db / 10)
        # snr_linear = 10 ** (self.snr_db / 10)
        noise_var = 1 / snr_linear  # công suất nhiễu

        # Ma trận pilot đơn vị (pilot_length x Nt)
        X = np.eye(self.pilot_length, self.Nt)

        for _ in range(self.num_samples):
            # Tạo kênh H (Nr x Nt) phức Gaussian Rayleigh
            H_real = np.random.normal(0, 1/np.sqrt(2), (self.Nr, self.Nt))
            H_imag = np.random.normal(0, 1/np.sqrt(2), (self.Nr, self.Nt))
            H = H_real + 1j * H_imag

            # Tạo nhiễu N (pilot_length x Nr) phức Gaussian
            N_real = np.random.normal(0, np.sqrt(noise_var/2), (self.pilot_length, self.Nr))
            N_imag = np.random.normal(0, np.sqrt(noise_var/2), (self.pilot_length, self.Nr))
            N = N_real + 1j * N_imag

            # Tín hiệu thu Y = X H^T + N
            # X shape: (pilot_length, Nt)
            # H shape: (Nr, Nt)
            # Y shape: (pilot_length, Nr)
            Y = X @ H.conj().T + N  # (pilot_length x Nr)

            # Chuyển sang định dạng real + imag cho input và output

            # # Input: Y (pilot_length x Nr) -> flatten -> real + imag
            # Y_real = Y.real.flatten()
            # Y_imag = Y.imag.flatten()
            # x_sample = np.concatenate([Y_real, Y_imag])  # vector chiều 2*pilot_length*Nr
            #
            # # Target: H (Nr x Nt) -> flatten -> real + imag
            # H_real = H.real.flatten()
            # H_imag = H.imag.flatten()
            # y_target = np.concatenate([H_real, H_imag])  # vector chiều 2*Nr*Nt

            # Input X_sample: Y (pilot_length x Nr) -> (2, pilot_length, Nr) (real, imag channels)
            Y_real = Y.real
            Y_imag = Y.imag
            x_sample = np.stack([Y_real, Y_imag], axis=0)  # Stack real and imag as channels (2, pilot_length, Nr)

            # Target Y_target: H (Nr x Nt) -> (2, Nr, Nt) (real, imag channels)
            H_real = H.real
            H_imag = H.imag
            y_target = np.stack([H_real, H_imag], axis=0)  # Stack real and imag as channels (2, Nr, Nt)

            x.append(x_sample)
            y.append(y_target)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
