# ThermalGuard Turbofan Maintenance Manual (Illustrative)

> Illustrative manual written for the ThermalGuard portfolio project. It is NOT OEM documentation and must
> not be used for real maintenance decisions. Sensor definitions follow the public NASA C-MAPSS documentation.

## 1. Sensor reference

| Channel | Symbol | Description | Unit |
|---|---|---|---|
| sensor_1 | T2 | Total temperature at fan inlet | °R |
| sensor_2 | T24 | Total temperature at LPC (low-pressure compressor) outlet | °R |
| sensor_3 | T30 | Total temperature at HPC (high-pressure compressor) outlet | °R |
| sensor_4 | T50 | Total temperature at LPT (low-pressure turbine) outlet | °R |
| sensor_5 | P2 | Pressure at fan inlet | psia |
| sensor_6 | P15 | Total pressure in bypass duct | psia |
| sensor_7 | P30 | Total pressure at HPC outlet | psia |
| sensor_8 | Nf | Physical fan speed | rpm |
| sensor_9 | Nc | Physical core speed | rpm |
| sensor_10 | epr | Engine pressure ratio (P50/P2) | ratio |
| sensor_11 | Ps30 | Static pressure at HPC outlet | psia |
| sensor_12 | phi | Ratio of fuel flow to Ps30 | pps/psi |
| sensor_13 | NRf | Corrected fan speed | rpm |
| sensor_14 | NRc | Corrected core speed | rpm |
| sensor_15 | BPR | Bypass ratio | ratio |
| sensor_16 | farB | Burner fuel-air ratio | ratio |
| sensor_17 | htBleed | Bleed enthalpy | – |
| sensor_18 | Nf_dmd | Demanded fan speed | rpm |
| sensor_19 | PCNfR_dmd | Demanded corrected fan speed | rpm |
| sensor_20 | W31 | HPT coolant bleed | lbm/s |
| sensor_21 | W32 | LPT coolant bleed | lbm/s |

Sensors 1, 5, 6, 10, 16, 18 and 19 are effectively constant under a single operating condition and carry no
degradation information. ThermalGuard models use the remaining 14 channels.

## 2. Degradation signatures

### 2.1 HPC degradation (fault mode in FD001 and FD002)
High-pressure compressor efficiency loss shows up as:
- **Rising temperatures** at the HPC outlet (T30, sensor_3) and the LPT outlet (T50, sensor_4).
- **Rising static pressure ratio indicators**: Ps30 (sensor_11) increases.
- **Falling fuel-flow ratio** phi (sensor_12) and falling coolant bleeds W31/W32 (sensor_20, sensor_21).
- Corrected core speed NRc (sensor_14) and physical core speed Nc (sensor_9) drift upward as the control
  system compensates.

Typical root causes: blade tip erosion, fouling of compressor stages, increased tip clearance.

### 2.2 Fan degradation (additional fault mode in FD003 and FD004)
Fan degradation shows as a change in bypass ratio BPR (sensor_15) and corrected fan speed NRf (sensor_13)
behaviour that differs from the HPC pattern. Because two fault modes are present, FD003/FD004 engines need
fault isolation before maintenance planning.

## 3. Health status and required actions

ThermalGuard converts the predicted remaining useful life (RUL, in operating cycles) into a status:

| Status | Predicted RUL | Required action |
|---|---|---|
| Healthy | ≥ 60 cycles | Continue normal operation. Review the trend weekly. |
| Warning | 30–59 cycles | Schedule a borescope inspection within 10 cycles. Order HPC blade kit. Increase monitoring to every cycle. |
| Critical | < 30 cycles | Plan removal at the next maintenance opportunity. Do not exceed 15 further cycles without engineering approval. |

An **anomaly alert** (unsupervised score above threshold for 3 consecutive cycles) means the sensor pattern
has left the healthy envelope. It does not mean failure is imminent. Treat an anomaly alert as a trigger for
an engineering review of the trend plots, even if the RUL status is still Healthy.

## 4. Inspection procedures

### 4.1 Borescope inspection of HPC
1. Confirm engine is shut down and cooled for at least 45 minutes.
2. Remove borescope access plugs for HPC stages 3, 5 and 7.
3. Inspect leading edges for erosion, nicks and coating loss; record tip clearance where accessible.
4. Classify findings: minor erosion (monitor), moderate erosion (re-inspect in 25 cycles), severe erosion or
   blade damage (removal).
5. Record findings in the maintenance log with engine id, cycle count and technician role token.

### 4.2 Compressor wash
A compressor water wash can recover 1–2% of HPC efficiency when degradation is caused by fouling rather than
erosion. Perform when T30 margin drops but the borescope shows no mechanical damage. After washing, the
ThermalGuard anomaly score typically drops; RUL predictions should be re-evaluated after 5 cycles of data.

### 4.3 Sensor validation
Before acting on a sudden change in a single channel, validate the sensor:
1. Compare the channel with physically related channels (e.g. T30 with Ps30 and T50).
2. A single channel jumping while related channels are steady indicates a sensor fault, not engine degradation.
3. Data rows with impossible values (negative temperatures, values above physical limits, missing channels)
   are quarantined automatically by the ThermalGuard data pipeline and never reach the models.

## 5. Operating conditions
FD002 and FD004 engines operate across six operating conditions defined by altitude, Mach number and throttle
resolver angle (op_setting_1..3). Sensor values must be compared within the same operating condition. Raw
values are not comparable across conditions without normalisation.

## 6. Data and model governance
- RUL predictions are decision support. A certified engineer approves every removal decision.
- Model versions are tracked in the MLflow registry; the model card for each version lists training data,
  metrics and limitations.
- Maintenance logs are pseudonymised: technician names are replaced with role tokens (e.g. TECH_07) before
  they are indexed by the maintenance assistant.
