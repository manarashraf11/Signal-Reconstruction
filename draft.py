import pandas as pd
import neurokit2 as nk

# Generate synthetic ECG signal
ecg = nk.ecg_simulate(duration=10, sampling_rate=260)

# Create a time vector for ECG
time_ecg = list(range(len(ecg)))

# Create a DataFrame for ECG
df_ecg = pd.DataFrame({
    'Time': time_ecg,
    'Amplitude': ecg
})

# Save the ECG DataFrame to a CSV file
df_ecg.to_csv('ecg_data.csv', index=False)

# Generate synthetic EEG signal
eeg = nk.eeg_simulate(duration=10, sampling_rate=46)

# Create a time vector for EEG
time_eeg = list(range(len(eeg)))

# Create a DataFrame for EEG
df_eeg = pd.DataFrame({
    'Time': time_eeg,
    'Amplitude': eeg
})

# Save the EEG DataFrame to a CSV file
df_eeg.to_csv('eeg_data.csv', index=False)
