import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CNN_MIMO
from dataset import MIMODataset
from pprint import pprint
from utils import collate_fn
from sklearn.metrics import r2_score


def reconstruct_complex_matrices(x_tensor, y_tensor, Nt, Nr, pilot_length):
    """
    Reconstructs complex Y_observed and H_true from real+imag tensors for CNN output.
    x_tensor: (batch, 2, pilot_length, Nr) - real/imag channels for Y_observed
    y_tensor: (batch, 2, Nr, Nt) - real/imag channels for H_true
    """
    # Y_observed_complex has shape (batch, pilot_length, Nr)
    Y_real = x_tensor[:, 0, :, :]  # Get real channel
    Y_imag = x_tensor[:, 1, :, :]  # Get imag channel
    Y_observed_complex = Y_real + 1j * Y_imag

    # H_true_complex has shape (batch, Nr, Nt)
    H_true_real = y_tensor[:, 0, :, :]  # Get real channel
    H_true_imag = y_tensor[:, 1, :, :]  # Get imag channel
    H_true_complex = H_true_real + 1j * H_true_imag

    return Y_observed_complex, H_true_complex


def reconstruct_H_mlp_complex(y_pred_tensor, Nr, Nt):
    """
    Reconstructs complex H_mlp from the CNN's real+imag output.
    y_pred_tensor: (batch, 2, Nr, Nt)
    """
    # H_mlp_complex has shape (batch, Nr, Nt)
    H_pred_real = y_pred_tensor[:, 0, :, :]
    H_pred_imag = y_pred_tensor[:, 1, :, :]
    H_mlp_complex = H_pred_real + 1j * H_pred_imag
    return H_mlp_complex


def nmse_complex(H_true, H_pred):
    """Computes Normalized Mean Squared Error for complex matrices."""
    # H_true, H_pred: (batch, Nr, Nt)
    # Ensure they are PyTorch tensors
    if not isinstance(H_true, torch.Tensor): H_true = torch.tensor(H_true)
    if not isinstance(H_pred, torch.Tensor): H_pred = torch.tensor(H_pred)

    error_power = torch.sum(torch.abs(H_true - H_pred) ** 2, dim=(1, 2))
    signal_power = torch.sum(torch.abs(H_true) ** 2, dim=(1, 2))
    nmse_val = torch.mean(error_power / (signal_power + 1e-9))  # Add epsilon for stability
    return nmse_val.item()


def mse_complex(H_true, H_pred):
    """Computes Mean Squared Error for complex matrices."""
    if not isinstance(H_true, torch.Tensor): H_true = torch.tensor(H_true)
    if not isinstance(H_pred, torch.Tensor): H_pred = torch.tensor(H_pred)

    # MSE is typically averaged over all elements, and then over the batch
    # Error for each channel matrix: sum(|H_true - H_pred|^2)
    # Total number of elements in each H: Nr * Nt
    # Avg over batch: mean(...)
    mse_val = torch.mean(torch.abs(H_true - H_pred) ** 2)
    return mse_val.item()


def r2_complex(H_true, H_pred):
    """Computes R2 score for complex matrices."""
    if not isinstance(H_true, torch.Tensor): H_true = torch.tensor(H_true)
    if not isinstance(H_pred, torch.Tensor): H_pred = torch.tensor(H_pred)

    # For R2 score, we need to flatten the complex numbers into real values
    # e.g., (batch, Nr*Nt*2) for H_true.real, H_true.imag, H_pred.real, H_pred.imag
    H_true_flat = torch.cat((H_true.real.reshape(H_true.size(0), -1), H_true.imag.reshape(H_true.size(0), -1)), dim=1)
    H_pred_flat = torch.cat((H_pred.real.reshape(H_pred.size(0), -1), H_pred.imag.reshape(H_pred.size(0), -1)), dim=1)

    # R2 score is calculated per sample, then averaged
    # Using sklearn's r2_score expects (n_samples, n_features)
    return r2_score(H_true_flat.cpu().numpy(), H_pred_flat.cpu().numpy())


# --- QPSK Modulation/Demodulation ---
QPSK_MAP = {
    (0, 0): (1 + 1j) / np.sqrt(2),
    (0, 1): (-1 + 1j) / np.sqrt(2),
    (1, 0): (1 - 1j) / np.sqrt(2),
    (1, 1): (-1 - 1j) / np.sqrt(2),
}
INV_QPSK_MAP = {v: k for k, v in QPSK_MAP.items()}


def qpsk_modulate(bits, Nt):
    """Modulates bits into QPSK symbols for Nt streams."""
    # bits: 1D numpy array of 0s and 1s
    num_bits = len(bits)
    assert num_bits % (2 * Nt) == 0, "Number of bits must be a multiple of 2*Nt for QPSK and Nt streams."
    symbols_per_stream = num_bits // (2 * Nt)

    modulated_streams = np.zeros((Nt, symbols_per_stream), dtype=np.complex64)
    bit_idx = 0
    for i in range(symbols_per_stream):
        for k in range(Nt):
            b1 = bits[bit_idx]
            b2 = bits[bit_idx + 1]
            modulated_streams[k, i] = QPSK_MAP[(b1, b2)]
            bit_idx += 2
    return modulated_streams  # Shape (Nt, num_symbols_per_stream)


def qpsk_demodulate(received_symbols):
    """Demodulates QPSK symbols (Nt x num_symbols) back to bits."""
    # received_symbols: (Nt, num_symbols_per_stream) numpy array
    Nt, num_symbols = received_symbols.shape
    demod_bits = []
    for i in range(num_symbols):
        for k in range(Nt):
            symbol = received_symbols[k, i]
            # Find closest QPSK symbol (hard decision)
            min_dist = float('inf')
            best_bits = (0, 0)
            for q_sym, bits_pair in INV_QPSK_MAP.items():
                dist = np.abs(symbol - q_sym) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_bits = bits_pair
            demod_bits.extend(best_bits)
    return np.array(demod_bits)


# --- Channel Estimation Methods ---

def ls_estimator(Y_observed_complex, X_pilot_complex):
    """
    Least Squares (LS) Channel Estimation.
    Y_observed_complex: (batch, pilot_length, Nr)
    X_pilot_complex: (pilot_length, Nt) - often identity matrix
    Returns H_ls_complex: (batch, Nr, Nt)
    """
    # For X_pilot_complex being identity (pilot_length = Nt, X = I_Nt),
    # Y = X H^T + N becomes Y = I H^T + N = H^T + N
    # So Y^H = H + N^H
    # H_ls = (X^H X)^-1 X^H Y. If X=I, H_ls = I^H Y = Y^H
    # Y_observed_complex is (batch, pilot_length, Nr)
    # H_ls_complex = Y_observed_complex.transpose(1, 2).conj()
    # Or, more generally:
    # (X^H X)^-1 @ X^H
    X_inv = torch.linalg.inv(X_pilot_complex.T.conj() @ X_pilot_complex) @ X_pilot_complex.T.conj()
    H_ls_complex = Y_observed_complex @ X_inv.T.conj()  # (batch, pilot_length, Nr) @ (Nr, pilot_length)
    # Need to verify dimensions carefully for batch processing.
    # A simpler way for X=I is H_ls = Y^H, as it is in problem context with X=I.
    # Since Y is (pilot_length, Nr) and pilot_length = Nt, we effectively have Y_complex (Nt, Nr)
    # H_ls_complex = Y_observed_complex.transpose(1, 2).conj()
    # Original H is (Nr, Nt).
    # If Y = H^T + N (X=I), then H_ls = Y^H = (H^T+N)^H = H+N^H
    # So H_ls_complex = Y_observed_complex.conj().permute(0, 2, 1) should result in (batch, Nr, Nt)
    return Y_observed_complex.conj().permute(0, 2, 1)


def mmse_estimator(Y_observed_complex, X_pilot_complex, noise_var_ch_est, Nt, Nr):
    """
    MMSE Channel Estimation.
    Assumes H is i.i.d complex Gaussian with variance 1 (i.e., E[HH^H]=I).
    Y_observed_complex: (batch, pilot_length, Nr)
    X_pilot_complex: (pilot_length, Nt)
    noise_var_ch_est: scalar (noise variance used during channel estimation)
    Returns H_mmse_complex: (batch, Nr, Nt)
    """
    # R_hh = E[H H^H] = I_Nt (assuming unit variance i.i.d Rayleigh fading for H)
    # R_nn = noise_var_ch_est * I_Nr
    # P_t = 1 (pilot transmit power is normalized to 1)
    # If X_pilot is Identity: Y = X H^T + N = I H^T + N = H^T + N
    # Then H_mmse = R_hh X^H (X R_hh X^H + R_nn)^-1 Y
    # = I I^H (I I^H + noise_var_ch_est I)^-1 Y
    # = (I + noise_var_ch_est I)^-1 Y
    # = (1 + noise_var_ch_est)^-1 Y
    # This formula applies if Y is a vector. For matrices and pilot = I:
    # H_mmse = (1.0 / (1.0 + noise_var_ch_est)) * H_ls
    # Assuming channel coefficients variance is 1 (power=1), H is (Nr, Nt)
    # pilot_length = Nt
    # X_pilot is identity (Nt, Nt)
    # Y is (Nt, Nr)
    # H_ls_complex = Y_observed_complex.conj().permute(0, 2, 1) # (batch, Nr, Nt)
    H_ls_complex = ls_estimator(Y_observed_complex, X_pilot_complex)
    mmse_factor = 1.0 / (1.0 + noise_var_ch_est)
    H_mmse_complex = mmse_factor * H_ls_complex
    return H_mmse_complex


# --- Data Detection Methods ---

def zf_detector(y_received_data_complex, H_est_complex):
    """
    Zero-Forcing (ZF) Detector.
    y_received_data_complex: (batch, Nr, num_symbols_per_stream)
    H_est_complex: (batch, Nr, Nt)
    Returns x_detected_complex: (batch, Nt, num_symbols_per_stream)
    """
    # x_hat = (H^H H)^-1 H^H y
    # For each batch sample:
    H_est_conj_trans = H_est_complex.transpose(1, 2).conj()  # (batch, Nt, Nr)
    H_H_H = H_est_conj_trans @ H_est_complex  # (batch, Nt, Nt)

    # Add a small diagonal term for stability (pseudo-inverse like behavior if H_H_H is singular)
    # This assumes H_H_H is invertible.
    # Pytorch's linalg.inv handles batches.
    H_H_H_inv = torch.linalg.inv(
        H_H_H + 1e-6 * torch.eye(H_H_H.shape[-1], device=H_H_H.device))  # Add epsilon for stability

    zf_filter = H_H_H_inv @ H_est_conj_trans  # (batch, Nt, Nr)
    x_detected_complex = zf_filter @ y_received_data_complex  # (batch, Nt, num_symbols_per_stream)
    return x_detected_complex


def mmse_data_detector(y_received_data_complex, H_est_complex, noise_var_data):
    """
    MMSE Data Detector.
    y_received_data_complex: (batch, Nr, num_symbols_per_stream)
    H_est_complex: (batch, Nr, Nt)
    noise_var_data: scalar (noise variance during data transmission)
    Returns x_detected_complex: (batch, Nt, num_symbols_per_stream)
    """
    # x_hat = (H^H H + N_0 I)^-1 H^H y
    # For each batch sample:
    H_est_conj_trans = H_est_complex.transpose(1, 2).conj()  # (batch, Nt, Nr)

    # (H^H H + N_0 I)
    term1 = H_est_conj_trans @ H_est_complex  # (batch, Nt, Nt)
    term2 = noise_var_data * torch.eye(H_est_complex.shape[-1], device=H_est_complex.device)  # (Nt, Nt)

    mmse_matrix = torch.linalg.inv(term1 + term2) @ H_est_conj_trans  # (batch, Nt, Nr)
    x_detected_complex = mmse_matrix @ y_received_data_complex  # (batch, Nt, num_symbols_per_stream)
    return x_detected_complex


# --- Evaluation Functions ---

def evaluate_channel_estimation(model, test_loader, device, Nt, Nr, pilot_length, pilot_matrix, snr_db_for_test_set):
    model.eval()
    mlp_nmse_values = []
    ls_nmse_values = []
    mmse_nmse_values = []

    mlp_mse_values = []
    ls_mse_values = []
    mmse_mse_values = []

    mlp_r2_values = []
    ls_r2_values = []
    mmse_r2_values = []

    with torch.no_grad():
        for x_batch, y_batch_true in tqdm(test_loader, desc="Evaluating Channel Est."):
            x_batch = x_batch.to(device)
            y_batch_true = y_batch_true.to(device)

            # Reconstruct complex Y_observed and H_true
            Y_observed_complex, H_true_complex = reconstruct_complex_matrices(x_batch, y_batch_true, Nt, Nr,
                                                                              pilot_length)

            # MLP Estimation
            y_pred_mlp = model(x_batch)
            H_mlp_complex = reconstruct_H_mlp_complex(y_pred_mlp, Nr, Nt)
            mlp_nmse_values.append(nmse_complex(H_true_complex, H_mlp_complex))
            mlp_mse_values.append(mse_complex(H_true_complex, H_mlp_complex))
            mlp_r2_values.append(r2_complex(H_true_complex, H_mlp_complex))

            # LS Estimation
            H_ls_complex = ls_estimator(Y_observed_complex, pilot_matrix)
            ls_nmse_values.append(nmse_complex(H_true_complex, H_ls_complex))
            ls_mse_values.append(mse_complex(H_true_complex, H_ls_complex))
            ls_r2_values.append(r2_complex(H_true_complex, H_ls_complex))

            # MMSE Estimation (requires noise variance for channel estimation)
            # We need to know the noise variance used when this test data was generated.
            # For a fair comparison, assume the SNR of the test data is known to the MMSE estimator.
            # This 'snr_db' for the test dataset is passed to the dataset class.
            # Let's assume the snr_db used for the test set is consistent.
            snr_linear_ch_est = 10 ** (snr_db_for_test_set / 10)
            noise_var_ch_est = 1 / snr_linear_ch_est
            H_mmse_complex = mmse_estimator(Y_observed_complex, pilot_matrix, noise_var_ch_est, Nt, Nr)
            mmse_nmse_values.append(nmse_complex(H_true_complex, H_mmse_complex))
            mmse_mse_values.append(mse_complex(H_true_complex, H_mmse_complex))
            mmse_r2_values.append(r2_complex(H_true_complex, H_mmse_complex))

    avg_mlp_nmse = np.mean(mlp_nmse_values)
    avg_ls_nmse = np.mean(ls_nmse_values)
    avg_mmse_nmse = np.mean(mmse_nmse_values)

    avg_mlp_mse = np.mean(mlp_mse_values)
    avg_ls_mse = np.mean(ls_mse_values)
    avg_mmse_mse = np.mean(mmse_mse_values)

    avg_mlp_r2 = np.mean(mlp_r2_values)
    avg_ls_r2 = np.mean(ls_r2_values)
    avg_mmse_r2 = np.mean(mmse_r2_values)

    print(f"\n--- Channel Estimation Results at {snr_db_for_test_set}dB SNR ---")
    print(f"{'Method':<10} {'NMSE (dB)':<12} {'MSE':<12} {'R2 Score':<12}")
    print(f"{'-' * 46}")
    print(f"{'CNN':<10} {10 * np.log10(avg_mlp_nmse):<12.4f} {avg_mlp_mse:<12.4f} {avg_mlp_r2:<12.4f}")
    print(f"{'LS':<10} {10 * np.log10(avg_ls_nmse):<12.4f} {avg_ls_mse:<12.4f} {avg_ls_r2:<12.4f}")
    print(f"{'MMSE':<10} {10 * np.log10(avg_mmse_nmse):<12.4f} {avg_mmse_mse:<12.4f} {avg_mmse_r2:<12.4f}")

    return {
        'CNN_NMSE': avg_mlp_nmse, 'CNN_MSE': avg_mlp_mse, 'CNN_R2': avg_mlp_r2,
        'LS_NMSE': avg_ls_nmse, 'LS_MSE': avg_ls_mse, 'LS_R2': avg_ls_r2,
        'MMSE_NMSE': avg_mmse_nmse, 'MMSE_MSE': avg_mmse_mse, 'MMSE_R2': avg_mmse_r2
    }


def simulate_transmission_ber(model, Nt, Nr, num_channels, snr_data_dbs, device, pilot_matrix, estimator_type='mlp'):
    model.eval()
    ber_results = {
        'Perfect CSI': [],
        'CNN Est.': [],
        'LS Est.': [],
        'MMSE Est.': []
    }

    # Use a fixed pilot matrix for all simulations
    # X_pilot = torch.tensor(np.eye(Nt, Nt), dtype=torch.complex64).to(device)

    for snr_db_data in tqdm(snr_data_dbs, desc=f"BER Sim. for {estimator_type}"):
        snr_linear_data = 10 ** (snr_db_data / 10)
        noise_var_data = 1 / snr_linear_data

        total_bits_perfect = 0
        error_bits_perfect = 0
        total_bits_mlp = 0
        error_bits_mlp = 0
        total_bits_ls = 0
        error_bits_ls = 0
        total_bits_mmse = 0
        error_bits_mmse = 0

        with torch.no_grad():
            for _ in range(num_channels):
                # 1. Generate a new channel (H) for each simulation instance
                H_real = np.random.normal(0, 1 / np.sqrt(2), (Nr, Nt))
                H_imag = np.random.normal(0, 1 / np.sqrt(2), (Nr, Nt))
                H_true = H_real + 1j * H_imag
                H_true_tensor = torch.tensor(H_true, dtype=torch.complex64).to(device)  # (Nr, Nt)

                # 2. Channel Estimation Phase (assuming pilot transmission and noise)
                # Create a sample for channel estimation input to MLP/LS/MMSE
                # This needs to be consistent with how dataset.py generates X.
                # Here, we generate Y_observed from a random noise realization.
                current_snr_db_ch_est = 10  # Assume channel estimation happens at a fixed SNR, e.g., 10dB
                snr_linear_ch_est = 10 ** (current_snr_db_ch_est / 10)
                noise_var_ch_est = 1 / snr_linear_ch_est

                N_ch_est_real = np.random.normal(0, np.sqrt(noise_var_ch_est / 2), (Nt, Nr))  # pilot_length x Nr
                N_ch_est_imag = np.random.normal(0, np.sqrt(noise_var_ch_est / 2), (Nt, Nr))
                N_ch_est = N_ch_est_real + 1j * N_ch_est_imag
                N_ch_est_tensor = torch.tensor(N_ch_est, dtype=torch.complex64).to(device)

                Y_observed_complex = pilot_matrix @ H_true_tensor.T.conj() + N_ch_est_tensor  # (Nt, Nr)

                # Prepare input for MLP (real/imag channels, 2D)
                Y_real_for_mlp = Y_observed_complex.real
                Y_imag_for_mlp = Y_observed_complex.imag
                x_for_mlp = torch.stack([Y_real_for_mlp, Y_imag_for_mlp], axis=0).unsqueeze(0).to(
                    device)  # (1, 2, Nt, Nr)

                # MLP Estimation
                y_pred_mlp_flat = model(x_for_mlp)  # (1, 2, Nr, Nt)
                H_mlp_est = reconstruct_H_mlp_complex(y_pred_mlp_flat, Nr, Nt).squeeze(0)  # (Nr, Nt)

                # LS Estimation
                H_ls_est = ls_estimator(Y_observed_complex.unsqueeze(0), pilot_matrix).squeeze(0)  # (Nr, Nt)

                # MMSE Estimation
                H_mmse_est = mmse_estimator(Y_observed_complex.unsqueeze(0), pilot_matrix, noise_var_ch_est, Nt,
                                            Nr).squeeze(0)  # (Nr, Nt)

                # 3. Data Transmission Phase
                # Number of data symbols to transmit per channel instance
                num_data_symbols_per_stream = 1000 // Nt  # Ensure enough bits for good BER stats
                num_bits_to_transmit = 2 * Nt * num_data_symbols_per_stream  # 2 bits per QPSK symbol per stream

                # Generate random bits for QPSK modulation
                original_bits = np.random.randint(0, 2, num_bits_to_transmit)
                # Modulate bits to QPSK symbols
                x_data_symbols = qpsk_modulate(original_bits, Nt)  # (Nt, num_symbols_per_stream)
                x_data_symbols_tensor = torch.tensor(x_data_symbols, dtype=torch.complex64).to(device)

                # Add noise for data transmission
                N_data_real = np.random.normal(0, np.sqrt(noise_var_data / 2), (Nr, num_data_symbols_per_stream))
                N_data_imag = np.random.normal(0, np.sqrt(noise_var_data / 2), (Nr, num_data_symbols_per_stream))
                N_data_tensor = torch.tensor(N_data_real + 1j * N_data_imag, dtype=torch.complex64).to(device)

                # Received signal at receiver: y_data = H x_data + N_data
                y_received_data_complex = H_true_tensor @ x_data_symbols_tensor + N_data_tensor  # (Nr, num_symbols_per_stream)

                # 4. Data Detection
                # Perfect CSI Detection
                x_perfect_detected = mmse_data_detector(y_received_data_complex.unsqueeze(0),
                                                        H_true_tensor.unsqueeze(0), noise_var_data).squeeze(
                    0)  # (Nt, num_symbols_per_stream)
                bits_perfect_demod = qpsk_demodulate(x_perfect_detected.cpu().numpy())

                # MLP Estimated Channel Detection
                x_mlp_detected = mmse_data_detector(y_received_data_complex.unsqueeze(0), H_mlp_est.unsqueeze(0),
                                                    noise_var_data).squeeze(0)
                bits_mlp_demod = qpsk_demodulate(x_mlp_detected.cpu().numpy())

                # LS Estimated Channel Detection
                x_ls_detected = mmse_data_detector(y_received_data_complex.unsqueeze(0), H_ls_est.unsqueeze(0),
                                                   noise_var_data).squeeze(0)
                bits_ls_demod = qpsk_demodulate(x_ls_detected.cpu().numpy())

                # MMSE Estimated Channel Detection
                x_mmse_detected = mmse_data_detector(y_received_data_complex.unsqueeze(0), H_mmse_est.unsqueeze(0),
                                                     noise_var_data).squeeze(0)
                bits_mmse_demod = qpsk_demodulate(x_mmse_detected.cpu().numpy())

                # 5. Calculate BER
                total_bits_perfect += len(original_bits)
                error_bits_perfect += np.sum(original_bits != bits_perfect_demod)

                total_bits_mlp += len(original_bits)
                error_bits_mlp += np.sum(original_bits != bits_mlp_demod)

                total_bits_ls += len(original_bits)
                error_bits_ls += np.sum(original_bits != bits_ls_demod)

                total_bits_mmse += len(original_bits)
                error_bits_mmse += np.sum(original_bits != bits_mmse_demod)

        ber_results['Perfect CSI'].append(error_bits_perfect / total_bits_perfect)
        ber_results['CNN Est.'].append(error_bits_mlp / total_bits_mlp)
        ber_results['LS Est.'].append(error_bits_ls / total_bits_ls)
        ber_results['MMSE Est.'].append(error_bits_mmse / total_bits_mmse)

    return ber_results


def get_arg():
    parser = argparse.ArgumentParser(description='Evaluate MIMO Channel Estimation and BER')
    parser.add_argument('--model_path', type=str, default="weights/best.pt", help="Path to CNN model checkpoint.")
    parser.add_argument('--Nt', type=int, default=2, help='Number of transmit antennas')
    parser.add_argument('--Nr', type=int, default=2, help='Number of receive antennas')
    parser.add_argument('--snr_db_channel_est', type=float, default=10, help='SNR (dB) for channel estimation phase')
    parser.add_argument('--num_test_samples', type=int, default=2000,
                        help='Number of samples for channel estimation testing')
    parser.add_argument('--num_ber_sim_channels', type=int, default=500,
                        help='Number of channel instances for BER simulation')
    parser.add_argument('--min_snr_ber', type=float, default=0, help='Minimum SNR (dB) for BER simulation')
    parser.add_argument('--max_snr_ber', type=float, default=20, help='Maximum SNR (dB) for BER simulation')
    parser.add_argument('--snr_step_ber', type=float, default=2, help='SNR (dB) step for BER simulation')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results (NMSE, MSE, R2, BER plot)')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the trained model
    model = CNN_MIMO(Nt=args.Nt, Nr=args.Nr).to(device)  # Initialize ResNetMIMO
    # print("\nCurrent model keys:")
    # pprint(model.state_dict().keys())
    # print("-" * 30)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    # print("\nCheckpoint keys:")
    # pprint(checkpoint["model"].keys())
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # Pilot matrix (identity, as used in dataset)
    pilot_matrix = torch.tensor(np.eye(args.Nt, args.Nt), dtype=torch.complex64).to(device)

    # --- 1. Evaluate Channel Estimation NMSE ---
    print("\n--- Evaluating Channel Estimation NMSE ---")
    # For testing, we use a fixed SNR for the dataset generation.
    test_dataset_params = {'num_samples': args.num_test_samples,
                           'snr_db_range': (args.snr_db_channel_est, args.snr_db_channel_est),  # Fixed SNR for testing
                           'Nt': args.Nt, 'Nr': args.Nr}
    test_dataset = MIMODataset(**test_dataset_params)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    print(f"Evaluating NMSE at SNR_DB for channel estimation: {args.snr_db_channel_est}")
    estimation_metrics = evaluate_channel_estimation(model, test_loader, device,
                                                     args.Nt, args.Nr, args.Nt,  # pilot_length = Nt
                                                     pilot_matrix,
                                                     args.snr_db_channel_est)

    # Save estimation metrics to a file
    estimation_results_file = os.path.join(args.output_dir, f"estimation_metrics_snr_{args.snr_db_channel_est}dB.txt")
    with open(estimation_results_file, 'w') as f:
        f.write(f"--- Channel Estimation Results at {args.snr_db_channel_est}dB SNR ---\n")
        f.write(f"{'Method':<10} {'NMSE (dB)':<12} {'MSE':<12} {'R2 Score':<12}\n")
        f.write(f"{'-' * 46}\n")
        f.write(
            f"{'CNN':<10} {10 * np.log10(estimation_metrics['CNN_NMSE']):<12.4f} {estimation_metrics['CNN_MSE']:<12.4f} {estimation_metrics['CNN_R2']:<12.4f}\n")
        f.write(
            f"{'LS':<10} {10 * np.log10(estimation_metrics['LS_NMSE']):<12.4f} {estimation_metrics['LS_MSE']:<12.4f} {estimation_metrics['LS_R2']:<12.4f}\n")
        f.write(
            f"{'MMSE':<10} {10 * np.log10(estimation_metrics['MMSE_NMSE']):<12.4f} {estimation_metrics['MMSE_MSE']:<12.4f} {estimation_metrics['MMSE_R2']:<12.4f}\n")
    print(f"Estimation metrics saved to {estimation_results_file}")

    # --- 2. Evaluate BER Performance ---
    print("\n--- Evaluating BER Performance ---")
    snr_data_dbs = np.arange(args.min_snr_ber, args.max_snr_ber + args.snr_step_ber, args.snr_step_ber)
    print(f"Simulating BER over SNR range: {args.min_snr_ber}dB to {args.max_snr_ber}dB")

    ber_results = simulate_transmission_ber(model, args.Nt, args.Nr,
                                            args.num_ber_sim_channels, snr_data_dbs,
                                            device, pilot_matrix, estimator_type='cnn')

    # Save BER results to a CSV file
    ber_results_file = os.path.join(args.output_dir, "ber_results.csv")
    with open(ber_results_file, 'w') as f:
        f.write("SNR_dB,Perfect_CSI,CNN_Est,LS_Est,MMSE_Est\n")
        for i, snr_db in enumerate(snr_data_dbs):
            f.write(f"{snr_db},{ber_results['Perfect CSI'][i]},")
            f.write(f"{ber_results['CNN Est.'][i]},{ber_results['LS Est.'][i]},")
            f.write(f"{ber_results['MMSE Est.'][i]}\n")
    print(f"BER results saved to {ber_results_file}")

    # Plotting BER results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_data_dbs, ber_results['Perfect CSI'], '-o', label='Perfect CSI')
    plt.semilogy(snr_data_dbs, ber_results['CNN Est.'], '-x', label='CNN Est.')  # Label as CNN Est.
    plt.semilogy(snr_data_dbs, ber_results['LS Est.'], '-s', label='LS Est.')
    plt.semilogy(snr_data_dbs, ber_results['MMSE Est.'], '-^', label='MMSE Est.')

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title(f'BER vs. SNR for MIMO ({args.Nr}x{args.Nt}) with QPSK', fontsize=14)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10)
    plt.ylim(1e-4, 1)  # Set y-axis limits to avoid too low BER
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plot_file = os.path.join(args.output_dir, "ber_plot.png")
    plt.savefig(plot_file)
    print(f"BER plot saved to {plot_file}")
    plt.show()


if __name__ == '__main__':
    args = get_arg()
    main(args)
