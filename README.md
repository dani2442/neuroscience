In this project we explore the dynamical systems of the brain and try to understand the mathematics behind it.

- [References](#references)
- [Data Modalities](#data-modalities)
- [Python packages](#python-packages)
- [Datasets](#datasets)
- [Exploratory Data Analysis: HCP Young Adult](#exploratory-data-analysis-hcp-young-adult)

# References

# Data Modalities

- MRI (Magnetic Resonance) - temporal ++ spatial resolution
  - Diffusion MRI (dMRI/DTI)
  - Functional MRI (fMRI)
  - resting-state fMRI (rfMRI)
  - task-fMRI (tfMRI)
- MEG (Magnetoencephalography): + temporal + spatial resolution
- EEG (Eloctroencephalography): + temporal - spatial resolution


# Python packages
- [nibabel](https://github.com/nipy/nibabel): read and write access to common neuroimaging file formats.
- [nilearn](https://github.com/nilearn/nilearn): versatile analyses of brain volumes and surfaces


# Datasets

| **Dataset**       | **Modalities**                      | **N (subjects)** | **Spatial Resolution**          | **Access**                           |
| :---------------- | :---------------------------------- | :--------------- | :------------------------------ | :----------------------------------- |
| HCP Young Adult   | T1/T2 MRI; DTI; resting/task fMRI   | \~1,200          | \~0.7 mm (T1), 2 mm fMRI        | Public (requires login/agreements)   |
| UK Biobank        | T1 MRI; T2 FLAIR; DTI; resting fMRI | \~100,000        | 1 mm (T1), 2–3 mm fMRI          | Public (application, approved users) |
| NKI-Rockland      | T1 MRI; DTI; resting fMRI           | \~1,000          | 1–2 mm (T1), 2–3 mm fMRI        | Public (INDI/NITRC; DUA)             |
| Cam-CAN           | T1 MRI; task/rest fMRI; MEG         | \~700            | 1 mm (T1); MEG \~5 mm eq. depth | Public (Cam-CAN repository)          |
| GSP (Superstruct) | T1 MRI; resting-state fMRI          | \~1,500          | \~1 mm (T1), 3.5 mm fMRI        | Restricted (Dataverse)               |


# Exploratory Data Analysis: HCP Young Adult
- Subjects: ~1,200 healthy adults
- Age Range: 22–35
- Family Structure: Includes twins and non-twin siblings
- Access: [ConnectomeDB](https://db.humanconnectome.org), requires user registration and agreement to data usage terms.

