import torch.nn as nn
import torch.nn.functional as F

class CNN_MIMO(nn.Module):
    def __init__(self, Nt=2, Nr=2):
        super().__init__()
        # Input shape: (batch_size, 2, Nt, Nr) for Y (real/imag channels)
        # Output shape: (batch_size, 2, Nr, Nt) for H (real/imag channels)

        # Encoder: giảm không gian feature map, tăng số channel
        # Input channel: 2 (real, imag)
        self.encoder = nn.Sequential(
            # (B, 2, Nt, Nr) -> (B, 32, Nt, Nr)
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (B, 32, Nt, Nr) -> (B, 64, Nt, Nr)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (B, 64, Nt, Nr) -> (B, 64, Nt/2, Nr/2) (if Nt, Nr are even)
            nn.MaxPool2d(2),
        )
        # Kích thước feature map sau encoder: (64, Nt/2, Nr/2)

        # Decoder: upsample trở lại kích thước Nt x Nr, giảm số channel về 2
        # Input channel: 64
        # Output channel: 2
        self.decoder = nn.Sequential(
            # (B, 64, Nt/2, Nr/2) -> (B, 32, Nt, Nr)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (B, 32, Nt, Nr) -> (B, 16, Nt, Nr)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # (B, 16, Nt, Nr) -> (B, 2, Nt, Nr)
            # Output will be of shape (batch_size, 2, Nt, Nr) for H.
            # We need (2, Nr, Nt) as target. This might need transpose.
            # Let's adjust for output (2, Nr, Nt)
            nn.Conv2d(16, 2, kernel_size=3, padding=1), # Output 2 channels (real/imag)
        )
        # Điều chỉnh output layer để đảm bảo kích thước cuối cùng là (2, Nr, Nt) nếu cần.
        # Hoặc thực hiện transpose ở hàm forward.
        self.Nt = Nt
        self.Nr = Nr


    def forward(self, x):
        # x_input shape: (batch_size, 2, pilot_length, Nr) -> (batch_size, 2, Nt, Nr) when pilot_length = Nt
        x = self.encoder(x)
        x = self.decoder(x)
        # x_output shape: (batch_size, 2, Nt, Nr)

        # Ta muốn output là (batch_size, 2, Nr, Nt) để khớp với H.
        # Cần kiểm tra xem có cần transpose Nt và Nr không.
        # Y là (pilot_length, Nr) -> input là (2, pilot_length, Nr)
        # H là (Nr, Nt) -> target là (2, Nr, Nt)
        # Nếu ta muốn encoder/decoder giữ nguyên kích thước không gian và chỉ đổi số kênh,
        # hoặc nếu H_true.conj().T là (Nt, Nr) thì có thể không cần transpose.
        # Trong dataset.py, Y = X @ H.conj().T + N. Với X=I, Y = H.conj().T + N
        # Do đó, Y_observed có shape (Nt, Nr).
        # H_true có shape (Nr, Nt).
        # Vậy input CNN là (2, Nt, Nr) và output CNN mong muốn là (2, Nr, Nt).

        # Nếu mô hình hiện tại output (2, Nt, Nr), ta cần transpose 2 chiều cuối cùng.
        # x = x.permute(0, 1, 3, 2) # (batch, channels, H, W) -> (batch, channels, W, H)
        return x