---
layout: post
title: Investigating Zebrafish Motor Behavior Alteration in a Lab-on-chip Model of Parkinson's Disease
image: "/posts/Panx1.jpg"
tags: [Data Analysis, Statistical Analysis, Behavioral Study, Neurodegenerative Mechanism Investigation]
---

In this project, I utilized advanced statistical processing and experimental design techniques to investigate neurodegenerative mechanisms in zebrafish larvae. Leveraging a lab-on-chip approach and a widely recognized model of PD (the 6-OHDA model), the project aimed to study behavioral and molecular responses, providing valuable insights into the early stages of neurodegeneration.

__

# Methods  <a name="data-overview"></a>

1. Behavioral screening 

The behavioral experiments were performed with a polydimethylsiloxane (PDMS) microfluidics device, complemented with key auxiliary components such as syringe pumps, a sourcemeter, and an upright Leica stereomicroscope with a camera to enable manipulation, stimulation, and imaging of zebrafish larvae.
During an experiment, a larva was transferred into the device and head-immobilized while its tail was free to move. Following a 1-minute recovery period, an electric stimulus was applied for 20 seconds using the sourcemeter, and the larva's locomotor response was recorded with a camera on the Leica stereomicroscope. The tested larva was removed from the device via the outlet before repeating the experiment to reach the designated sample sizes for each condition.

2. Video and Image Analysis: 

For analyzing tail movement, the point tracking feature in the open-source Kinovea software (www.kinovea.org, France) was utilized. Each video frame was checked to ensure that the software was accurately tracking the tail tip. 
In low-resolution frames where the tail tip was blurred as a result of fast movement, we manually adjusted or selected the tip position. The tail position data was saved in.xml format and the electric-induced response duration (RD) and tail beat frequency (TBF) phenotypes were analyzed in Microsoft Excel (Microsoft Corp., WA, USA). 
RD was defined as the time from the start of tail movement until the zebrafish activities ceased. The start time was at the point that the electric current was applied because there was no delay in response to the stimulus. 
TBF represents the number of full tail movement cycles divided by the total RD, with small tail flicks excluded. The tail was required to pass a threshold of ± 0.25 mm from the centerline to produce a quarter-strike cycle for each pass. Therefore, four quarter strikes resulted in a complete cycle with bilateral and unilateral tail turns equated to one and a half cycles, respectively.

3. Statistical Processing:
   
Applied common statistical methods to understand data distribution and identify significant differences.
Utilized the Shapiro–Wilk test to assess the normality of distribution for behavioral data.
Employed a Mann–Whitney test for non-normally distributed data.
Used a Pair Wise Fixed Reallocation Randomisation Test for statistical significance in RT-qPCR data.

4. Sample Size Determination:

Utilized G*Power 3.1 software for power analysis to determine sample sizes.
Considered an effect size of 0.5, a significance level of 0.05, and a power of 80%.

__

# Result  <a name="data-overview"></a>

A lab-on-chip approach was used to test the hypothesis that loss of Panx1 function alters an experimental PD phenotype. 
We propose that zebrafish Panx1a models offer opportunities to shed light on PD's physiological and molecular basis. Panx1a might play a role in the progression of PD, and therefore deserves further investigation.

__

# Conclusion  <a name="data-overview"></a>

Currently, there is no standard treatment for PD; Therefore, finding relevant factors underlying the pathophysiological progression of this disease is required for translational research addressing this severe health problem. Starting this discovery process early in life before the typical clinical symptoms of PD manifest would be the most likely approach toward prevention. 
In summary, the key findings of this study have the potential to foster new lines of research, representing a view into the earliest insults driving a vertebrate toward PD.

Further information: https://onlinelibrary.wiley.com/doi/full/10.1002/jnr.25241
