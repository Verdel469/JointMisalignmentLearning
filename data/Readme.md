# A dataset to investigate the prediction of upper limb torque from EMG signals
## Motivation
EMG-driven exoskeleton assistance requires the use of intention detection models to associate electromyographic signals with some feature of human movement, such as angular position, velocity, or joint torque. The goal of this dataset is to provide data that allows the benchmarking of such models for a variety of movements.

## Short description
This dataset includes kinematic, dynamic and electromyographic data from 17 participants (11 males, age 28.2 ± 7 years, height 175.4 ± 7 cm, weight 70 ± 11 kg). These data were collected during the performance of a sagittal plane upper limb tracking task for single joint (elbow flexion/extension) and multiple joint (elbow and shoulder flexion/extension).

## Methodology
A detailed description of the data collection methodology can be found here: https://www.biorxiv.org/content/10.1101/2024.01.11.575155v1

## Data Description
The data set consists of 17 folders, one for each participant. Inside each folder you will find
- A metadata file (**SXX.json**) containing information about the subject: age, sex, weight, height, and upper limb masses and lengths.
- An OpenSim model file (**scaledModel.osim**) containing a scaled upper limb model of the given subject.
- A **MVC** folder containing EMG data from maximal voluntary contraction trials
- A **SJ** folder containing trial folders for the single joint condition (elbow flexion/extension).
- A **MJ** folder containing test folders for the multi-joint condition (elbow and shoulder flexion/extension)

### EMG Data
The files containing EMG data have the following header
``` 
TIME,DELTAnt,DELTMed,DELTPost,PECT,LATI,TRILong,TRILat,TRIMed,BICLong,BICShort,BRA,BRD
```
This corresponds to a time stamp and EMG signals from the anterior, median and posterior detloids, pectoralis major, latissimus dorsi, long, lateral and median triceps, long and short biceps, brachioradialis and brachialis.

### MVC data
The MVC folder contains two files: **emgMVCElbow.csv** and **emgMVCShoulder.csv**. They were collected during the realisation of maximum voluntary contraction tasks and contain raw EMG data sampled at 2 kHz.

### Trial data
#### Single-joint condition
Single-joint trials contain 5 files:
- **emgFilt.csv**: Filtered EMG signals, using a 20-450 Hz bandpass filter, a rectification, a 3Hz lowpass filter and normalized with MVC. Sampled at 100 Hz.
- **emgRaw.csv**: Raw EMG signals. Sampled at 2 kHz.
- **humanPositions.csv**: Angular position of the elbow in rad. Sampled at 100 Hz.
- **humanVelocities.csv**: Angular velocity of the elbow in rad/s. Sampled at 100 Hz.
- **muscleTorque.csv**: Joint torque of the elbow in N.m. Sampled at 100 Hz.
  
#### Multi-joint condition
Multi-joint trials contain 5 files:
- **emgFilt.csv**: Filtered EMG signals, using a 20-450 Hz bandpass filter, a rectification, a 3Hz lowpass filter and normalized with MVC. Sampled at 100 Hz.
- **emgRaw.csv**: Raw EMG signals. Sampled at 2 kHz.
- **humanPositions.csv**: Angular positions of the upper limb in rad. Sampled at 100 Hz. 
- **humanVelocities.csv**: Angular velocities of the upper limb rad/s. Sampled at 100 Hz.
- **muscleTorque.csv**: Joint torque of the upper limb in N.m. Sampled at 100 Hz.

For this trial, kinematic and torque files use the following headers:
```
TIME,elv_angle,shoulder_elv,shoulder_rot,elbow_flexion,pro_sup,deviation,flexion
```
The columns correspond to the OpenSim model coordinates. For sagittal plane movement, columns of interest are **shoulder_elv** for shoulder flexion/extension and **elbow_flexion** for elbow flexion/extension.