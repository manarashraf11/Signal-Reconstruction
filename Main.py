import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets
from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq
from scipy.signal import find_peaks

import Classes
from Classes import FileBrowser
import GUI
import sys

class MainApp(QtWidgets.QMainWindow, GUI.Ui_MainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.fileBrowser = FileBrowser(self)
        self.numOfGeneratedSins = 0
        self.signals = []  # List to store Signal objects
        self.originalSignalAmplitudes = np.array([])
        self.noisySignalAmplitudes = np.array([])
        self.previous_scatter = None
        self.isLoaded = False
        self.time = np.linspace(0, 1, 1000)  # Create 1000 evenly spaced time points
        self.csvFilesCount = 1
        
        self.pushButton_next.clicked.connect(self.getToTab2)
        self.pushButton_clear.clicked.connect(self.clearComposer)
        self.pushButton_add.clicked.connect(self.generateSinusoidal)
        self.pushButton_delete.clicked.connect(self.removeSinusoidal)
        self.pushButton_export.clicked.connect(self.exportComposedSignalToCsv)
        self.pushButton_clear_all.clicked.connect(self.clearAll)

        self.plotwidget_error.setYRange(1.2, -1.2)

        self.horizontalSlider_freq.setMinimum(1)
        self.horizontalSlider_freq.setMaximum(100)
        self.horizontalSlider_freq.setSingleStep(1)

        self.horizontalSlider_amplitude.setMinimum(1)
        self.horizontalSlider_amplitude.setMaximum(100)
        self.horizontalSlider_amplitude.setSingleStep(1)

        self.horizontalSlider_phase.setMinimum(0)
        self.horizontalSlider_phase.setMaximum(360)
        self.horizontalSlider_phase.setSingleStep(1)

        self.horizontalSlider_SNR.setMinimum(0)
        self.horizontalSlider_SNR.setMaximum(30)
        self.horizontalSlider_SNR.setSingleStep(1)

        self.horizontalSlider_freq.valueChanged.connect(self.previewSinusoidal)
        self.horizontalSlider_amplitude.valueChanged.connect(self.previewSinusoidal)
        self.horizontalSlider_phase.valueChanged.connect(self.previewSinusoidal)

        self.horizontalSlider_SNR.valueChanged.connect(self.addNoise)
        self.horizontalSlider_fs.valueChanged.connect(self.sampleSignal)

        self.checkBox_noise.stateChanged.connect(self.checkForNoiseAddition)

        self.radioButton_normalized.setChecked(True)

        self.radioButton_normalized.toggled.connect(lambda checked: (self.sampleSignal(), self.updateSliderLimits()))
        self.radioButton_actual.toggled.connect(lambda checked: (self.sampleSignal(), self.updateSliderLimits()))

        self.comboBox_signals.activated.connect(self.viewSelectedSinusoidal)
        self.actionOpen.triggered.connect(self.loadSignal)

        # Initialize the slider limits based on the initial state of the radio buttons
        self.updateSliderLimits()

    def previewSinusoidal(self):
        self.plotwidget_mixer.clear()
        # get the variables values from sliders
        frequency = self.horizontalSlider_freq.value()
        amplitude = self.horizontalSlider_amplitude.value()
        phase_degrees = self.horizontalSlider_phase.value()
        # Convert phase to radians
        phase_radians = np.deg2rad(phase_degrees)  
        # calculate y values of the signal, returns (1000,) nd array
        signalYCoordinates = amplitude * np.sin(2 * np.pi * frequency * (self.time) + phase_radians)
        print(len(signalYCoordinates))
        self.plotwidget_mixer.setLimits(xMin=self.time[0] - 0.1, xMax=self.time[-1] + 0.1, yMin=min(signalYCoordinates) - 0.1, yMax=max(signalYCoordinates) + 0.1)
        self.plotwidget_mixer.plot(self.time, signalYCoordinates)
        return frequency, amplitude, phase_degrees, signalYCoordinates
    
    def viewSelectedSinusoidal(self):
        # get the current selected signal
        selectedSignalName = self.comboBox_signals.currentText()
        # get the sinusoidal object by its name
        selectedSignal = [signal for signal in self.signals if signal.signalName == selectedSignalName]
        # set its parameters values to sliders
        if selectedSignal:
            self.horizontalSlider_freq.setValue(selectedSignal[0].frequency)
            self.horizontalSlider_amplitude.setValue(selectedSignal[0].amplitude)
            self.horizontalSlider_phase.setValue(selectedSignal[0].phase)

    def generateSinusoidal(self):
        frequency, amplitude, phase_degrees, signalYCoordinates = self.previewSinusoidal()
        signalName = 'Signal {}'.format(self.numOfGeneratedSins + 1)
        # create and add a new sinusoidal object
        signal = Classes.Sinusoidal(signalName, frequency, amplitude, self.time, phase_degrees, signalYCoordinates)
        self.signals.append(signal)
        self.numOfGeneratedSins += 1
        self.comboBox_signals.addItem(signalName)
        # clear preview widget and reset sliders
        self.plotwidget_mixer.clear()
        self.horizontalSlider_freq.setValue(0)
        self.horizontalSlider_amplitude.setValue(0)
        self.horizontalSlider_phase.setValue(0)
        # replotting the composed signal after adding one item
        self.displayComposedSignal()

    def removeSinusoidal(self):
        if len(self.signals) > 1:
            # get the selected signal name to remove
            signalName = self.comboBox_signals.currentText()
            # get the selected signal index to remove
            indexToRemove = self.comboBox_signals.findText(signalName)
            # remove signal from comboBox_signals and signals list
            if indexToRemove != -1:
                # remove sinusoid
                self.comboBox_signals.removeItem(indexToRemove)
                self.signals = [signal for signal in self.signals if signal.signalName != signalName]
                self.numOfGeneratedSins -= 1
                # clear preview widget and reset sliders
                self.plotwidget_mixer.clear()
                self.horizontalSlider_freq.setValue(0)
                self.horizontalSlider_amplitude.setValue(0)
                self.horizontalSlider_phase.setValue(0)
            # replotting the composed signal after removing one item
            self.displayComposedSignal()
        else:
            self.clearComposer()
    
    def displayComposedSignal(self):
        # Remove the signal with name "loaded_signal" from the self.signals list
        self.signals = [signal for signal in self.signals if signal.signalName != "loaded_signal"]
        # Initialize with zeros
        total_signalY = np.zeros(self.time.shape)
        # get magnitude sum of all signals
        for signal in self.signals:
            print(len(signal.signalYCoordinates ))
            total_signalY += signal.signalYCoordinates 

        self.plotwidget_result.setLimits(xMin=self.time[0] - 0.1, xMax=self.time[-1] + 0.1,
                                         yMin=min(total_signalY) - 0.1, yMax=max(total_signalY) + 0.1)
        self.plotwidget_result.clear()
        # plot composed signal
        self.plotwidget_result.plot(self.time, total_signalY)
        return total_signalY

    def clearComposer(self):
        # clear widgets
        self.plotwidget_result.clear()
        self.plotwidget_mixer.clear()
        # clear comboBox_signals
        self.comboBox_signals.clear()
        # reset variables
        self.numOfGeneratedSins = 0
        self.signals = []
        # reset sliders
        self.horizontalSlider_freq.setValue(0)
        self.horizontalSlider_amplitude.setValue(0)
        self.horizontalSlider_phase.setValue(0)

    def exportComposedSignalToCsv(self):
        composedSignalAmplitudes = self.displayComposedSignal()
        data = {
            "Time": self.time,
            "Amplitude": composedSignalAmplitudes
        }
        # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Specify the filename
        frequencyComponents = []
        for signal in self.signals:
            frequencyComponents.append(str(signal.frequency))

        filename = f"ComposedSignal{self.csvFilesCount}_{('_'.join(frequencyComponents))}.csv"

        # Export the DataFrame to a CSV file
        df.to_csv(filename, index=False)

############### Tab 2 ####################

    def getToTab2(self):
        plotItem = self.plotwidget_result.getPlotItem()
        curves = plotItem.listDataItems()
        if curves:
            self.isLoaded = False
            self.tabWidget.setCurrentIndex(1)
            self.clearAll()
            signalAmplitudeValues = self.displayComposedSignal()
            self.originalSignalAmplitudes = np.array([])
            self.originalSignalAmplitudes = np.append(self.originalSignalAmplitudes, signalAmplitudeValues)
            self.plotOriginalSignal()
        else:
            self.tabWidget.setCurrentIndex(1)

    def loadSignal(self):
        # Remove the signal with name "loaded_signal" from the self.signals list
        self.signals = [signal for signal in self.signals if signal.signalName != "loaded_signal"]
        self.isLoaded = True
        time, amplitude = self.fileBrowser.browse_file()
        loaded_signal = Classes.Sinusoidal("loaded_signal", 0, 0, time, 0, amplitude)
        self.signals.append(loaded_signal)
        self.originalSignalAmplitudes = np.array([])
        self.originalSignalAmplitudes = np.append(self.originalSignalAmplitudes, amplitude[:1000])
        self.plotOriginalSignal()

    def plotOriginalSignal(self):
        self.plotwidget_original.clear()
        self.tabWidget.setCurrentIndex(1)
        # Add a legend to the plot
        self.plotwidget_original.addLegend()
        # Plot the signal and add it to the legend
        self.plotwidget_original.plot(self.time, self.originalSignalAmplitudes, name="Max Frequency: " + str(self.getMaxFrequency()))
        self.plotwidget_original.setLimits(xMin=self.time[0] - 0.1, xMax=self.time[-1] + 0.1,
                                           yMin=min(self.originalSignalAmplitudes) - 0.1, yMax=max(self.originalSignalAmplitudes) + 0.1)

    def sampleSignal(self):
        self.plotwidget_original.clear()

        if not np.any(self.noisySignalAmplitudes):
            signalAmplitudeValues = self.originalSignalAmplitudes
        else:
            signalAmplitudeValues = self.noisySignalAmplitudes

        #### sample #####
        fMax = self.getMaxFrequency()

        # Adjust the sampling frequency based on which radio button is checked
        if self.radioButton_actual.isChecked():
            # self.horizontalSlider_fs.setEnabled(True)
            # Get the value of the slider
            sliderValue = self.horizontalSlider_fs.value()
            
            samplingFrequency = sliderValue
        else:
            # Get the value of the slider
            sliderValue = self.horizontalSlider_fs.value()
            samplingFrequency = sliderValue * fMax

        # Sample the signal using the specified sampling frequency
        samplingT = 1 / samplingFrequency
        samplesTimeValues = np.arange(0, 1, step=samplingT)

        # Find the corresponding indices in the 'time' array
        indices = np.searchsorted(self.time, samplesTimeValues)

        # Retrieve the corresponding amplitudes from 'signalAmplitudeValues'
        samplesAmpValues = signalAmplitudeValues[indices]

        ####### reconstruct ##########

        # Apply Whittaker–Shannon (sinc) interpolation to reconstruct the signal
        reconstructedSignal = self.whittakerShannonInterpolation(samplesAmpValues, samplesTimeValues, len(self.time), samplingT)

        # Calculate the error signal
        errorSignal = signalAmplitudeValues - reconstructedSignal

        ####### Display #######

        # clear plots before replotting
        self.plotwidget_error.clear()
        self.plotwidget_reconstructed.clear()
        
        # Remove the previous scatter plot item if it exists
        if self.previous_scatter is not None:
            self.plotwidget_original.removeItem(self.previous_scatter)

        # Add a legend to the plot
        self.plotwidget_original.addLegend()
        # Plot the original signal with markers, the reconstructed signal, and the error signal
        self.plotwidget_original.plot(self.time, signalAmplitudeValues, title="Original ECG Signal", name="Max Frequency: " + str(self.getMaxFrequency()))
        
        # Add sampling point markers
        self.plot_scatter = pg.ScatterPlotItem(pos=zip(samplesTimeValues, samplesAmpValues), symbol='o', size=10, brush='b')
        self.plotwidget_original.addItem(self.plot_scatter)
        self.previous_scatter = self.plot_scatter
        
        # plot reconstructed signal
        self.plotwidget_reconstructed.plot(self.time, reconstructedSignal)
        
        # plot error signal
        self.plotwidget_error.plot(self.time, errorSignal)

        # set plotwidgets limit
        self.plotwidget_original.setLimits(xMin=self.time[0] - 0.1, xMax=self.time[-1] + 0.1, yMin=min(signalAmplitudeValues) - 0.1, yMax=max(signalAmplitudeValues) + 0.1)
        self.plotwidget_reconstructed.setLimits(xMin=self.time[0] - 0.1, xMax=self.time[-1] + 0.1, yMin=min(reconstructedSignal) - 0.5, yMax=max(reconstructedSignal) + 0.5)
        self.plotwidget_reconstructed.setYRange(min(reconstructedSignal - 0.1), max(reconstructedSignal + 0.1))
        self.plotwidget_error.setYRange(8.2, -8.2)
        self.plotwidget_error.setLimits(xMin=self.time[0] - 0.1, xMax=self.time[-1] + 0.1, yMin=-8.2, yMax=8.2)
        
    def addNoise(self, SNR):
        if self.checkBox_noise.isChecked():
            amplitude = self.originalSignalAmplitudes

            # Calculate the variance of the noise based on SNR
            # The variance of the noise can be calculated from the signal power and the SNR
            signal_power = np.mean(amplitude ** 2)
            noise_variance = signal_power / (10 ** (SNR / 10))

            # Generate random noise with the same length as the original signal
            noise = np.random.normal(0, np.sqrt(noise_variance), len(amplitude))
            # noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)

            # Add the noise to the original signal
            self.noisySignalAmplitudes = amplitude + noise
        else:
            self.noisySignalAmplitudes = np.array([])

        self.sampleSignal()

    def checkForNoiseAddition(self):
        if self.checkBox_noise.isChecked():
            self.addNoise(self.horizontalSlider_SNR.value())
        else:
            self.noisySignalAmplitudes = np.array([])
            self.sampleSignal()

    def whittakerShannonInterpolation(self, samplesAmpValues, samplesTimeValues, output_length, samplingPeriod):
        reconstructed_signal = np.zeros(output_length)
        
        for i, amp in enumerate(samplesAmpValues):
            # Calculate the sinc function for all time values in one step
            sinc_values = np.sinc((self.time - samplesTimeValues[i]) / samplingPeriod)
            # Accumulate the contribution of each sample to the reconstructed signal
            reconstructed_signal += amp * sinc_values
        return reconstructed_signal

    def getMaxFrequency(self):
        if not self.isLoaded:
            # Get the maximum frequency value from the list
            maxFrequency = max(self.signals, key = lambda signal: signal.frequency).frequency
            return maxFrequency
        else:
            loaded_signal = next((signal for signal in self.signals if signal.signalName == "loaded_signal"), None)
            time = loaded_signal.time
            Ts = time[2] - time[1]
            Fs = 1 / Ts
            print(Fs)
            max_frequency = Fs / 2
            return max_frequency

            # # Compute the FFT of your signal
            # composed_fft = fft(self.originalSignalAmplitudes)
            # # Compute the magnitude spectrum
            # magnitude_spectrum = np.abs(self.originalSignalAmplitudes)
            # # Compute the corresponding frequencies
            # frequencies = fftfreq(len(magnitude_spectrum), 1.0/360.0)
            # # Find the index of the peak in the magnitude spectrum
            # peak_indices, _ = find_peaks(magnitude_spectrum)
            # # Get the frequency of the peak
            # peak_frequencies = frequencies[peak_indices]
            # max_frequency = max(peak_frequencies)
            # return max_frequency

    def updateSliderLimits(self):
        if self.radioButton_actual.isChecked():
            self.horizontalSlider_fs.setRange(1, 1000)
            self.horizontalSlider_fs.setValue(self.horizontalSlider_fs.minimum())
        elif self.radioButton_normalized.isChecked():
            self.horizontalSlider_fs.setRange(1, 10)
            self.horizontalSlider_fs.setValue(self.horizontalSlider_fs.minimum())

    def clearAll(self):
        # clear all amplitude lists
        self.originalSignalAmplitudes = np.array([])
        self.noisySignalAmplitudes = np.array([])

        # Clear all the plots
        self.plotwidget_original.clear()
        self.plotwidget_reconstructed.clear()
        self.plotwidget_error.clear()

        # Reset the sliders to their minimum values
        self.horizontalSlider_fs.setValue(self.horizontalSlider_fs.minimum())
        self.horizontalSlider_SNR.setValue(self.horizontalSlider_SNR.minimum())

        self.radioButton_normalized.setChecked(True)
        self.checkBox_noise.setChecked(False)
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())