import h5py
import numpy as np
import scipy.ndimage
from torch.utils.data import Dataset

    
###############################
# 1) DATA LOADING & RESIZING
###############################
def load_swept_sine_case(file_path, amplitude_key, target_x, target_y):
    """Redimensionné en 48x48 pour être divisible par 16 (patch size)"""
    with h5py.File(file_path, 'r') as f:
        dataset = f['data_structure']['swept_sines'][amplitude_key]
        wz_grid = np.array(dataset['wz_grid'])  # (6405, T)
        T = wz_grid.shape[1]
        spatial_x, spatial_y = 61, 105
        wz_3d = wz_grid.reshape(spatial_x, spatial_y, T)
        u = np.array(dataset['y'])
        if u.ndim > 1:
            u = u.squeeze()
        frames_resized = np.zeros((target_x, target_y, T), dtype=np.float32)
        for t in range(T):
            frames_resized[:, :, t] = scipy.ndimage.zoom(
                wz_3d[:, :, t],
                (target_x / spatial_x, target_y / spatial_y),
                order=1
            )
        frames_resized = np.transpose(frames_resized, (2, 0, 1))  # (T, target_x, target_y)
        frames_resized = frames_resized[:, np.newaxis, :, :]      # (T, 1, target_x, target_y)
        return {
            'amplitude': amplitude_key,
            'u': u.astype(np.float32),
            'frames': frames_resized
        }

def load_all_amplitudes(file_path, amplitude_map, amplitude_list, img_size):
    data_list = []
    for amp in amplitude_list:
        amp_key = amplitude_map[amp]
        data_case = load_swept_sine_case(file_path, amp_key, target_x=img_size, target_y=img_size)
        data_list.append(data_case)
    return data_list

def split_data(file_path, img_size):
    amplitude_map = {
        0.5:  'A0p05',
        0.75: 'A0p075',
        1.0:  'A0p10',
        1.25: 'A0p125',
        1.5:  'A0p15',
        1.75: 'A0p175',
        2.0:  'A0p20',
        2.25: 'A0p225',
        2.5:  'A0p25',
        2.75: 'A0p275',
        3.0:  'A0p30'
    }
    train_amps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    val_amps   = [0.75, 1.75, 2.75]
    test_amps  = [1.25, 2.25]
    train_list = load_all_amplitudes(file_path, amplitude_map, train_amps, img_size)
    val_list   = load_all_amplitudes(file_path, amplitude_map, val_amps, img_size)
    test_list  = load_all_amplitudes(file_path, amplitude_map, test_amps, img_size)
    return train_list, val_list, test_list

###############################
# 1.1) NORMALISATION DES DONNÉES
###############################
def compute_normalization_stats(data_list):
    all_frames = np.concatenate([data['frames'] for data in data_list], axis=0)
    frame_mean = all_frames.mean()
    frame_std = all_frames.std()
    all_u = np.concatenate([data['u'] for data in data_list], axis=0)
    u_mean = all_u.mean()
    u_std = all_u.std()
    return frame_mean, frame_std, u_mean, u_std

def normalize_data_list(data_list, frame_mean, frame_std, u_mean, u_std):
    for data in data_list:
        data['frames'] = (data['frames'] - frame_mean) / frame_std
        data['u'] = (data['u'] - u_mean) / u_std

###############################
# 2) CREATING SEQUENCES FOR JiT
###############################
def create_jit_sequences(data_list, past_window=2):
    """Créer des séquences pour l'entraînement avec JiT"""
    X_frames_list, X_u_past_list, X_u_curr_list, Y_list = [], [], [], []
    for data_case in data_list:
        frames = data_case['frames']  # (T, 1, H, W)
        u = data_case['u']            # (T,)
        T = frames.shape[0]
        for i in range(past_window, T):
            past_f = frames[i-past_window:i]          # (past_window, 1, H, W)
            past_u = u[i-past_window:i].reshape(-1, 1)
            current_u = np.array([u[i]], dtype=np.float32)
            target_f = frames[i]                       # (1, H, W)
            X_frames_list.append(past_f)
            X_u_past_list.append(past_u)
            X_u_curr_list.append(current_u)
            Y_list.append(target_f)
    X_frames = np.array(X_frames_list, dtype=np.float32)
    X_u_past = np.array(X_u_past_list, dtype=np.float32)
    X_u_curr = np.array(X_u_curr_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X_frames, X_u_past, X_u_curr, Y

class FluidDynamicsDataset(Dataset):
    def __init__(self, X_frames, X_u_past, X_u_curr, Y):
        self.X_frames = X_frames
        self.X_u_past = X_u_past
        self.X_u_curr = X_u_curr
        self.Y = Y
    
    def __len__(self):
        return len(self.X_frames)
    
    def __getitem__(self, idx):
        return self.X_frames[idx], self.X_u_past[idx], self.X_u_curr[idx], self.Y[idx]