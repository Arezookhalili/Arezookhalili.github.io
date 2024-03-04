---
layout: post
title: Computer-Aided Analysis of Electric-Induced Tail Patterns in Zebrafish Larvae
image: "/posts/Movement_Pattern.png"
tags: [Data Analysis, Statistical Analysis, Computer-Aided Movement Pattern Detection]
---

This project introduces an innovative approach for analyzing electric-induced behavioral responses of zebrafish larvae using a microfluidic device and a MATLAB script. The methodology allows the extraction of specific tail movement patterns, such as J- and C-bends, providing a deeper understanding of how electric stimuli modulate zebrafish locomotion. This technique holds significance for diverse biological applications, shedding light on sensory-motor systems and neural circuit involvement in generating distinct movement patterns.

__

# Methodology  <a name="data-overview"></a>

The point tracking software, Kinovea (https://www.kinovea.org/), was used to provide the position data of three red points relative to the origin along the tail (Figure b). A MATLAB script was then run to determine response duration (RD) and tail beat frequency (TBF) as well as the number of pattern repetitions with more caudal and rostral bends classified as J- and C-like bend (slow and burst swimming) patterns, respectively (Figure c and d). The script applied several conditions and thresholds based on the angles between the tracked points on the tail to binarize the data and discern the presence of a pattern of interest.  

### 1. Data Collection:

Gathered data on electric-induced zebrafish larvae' tail movement.

### 2. Video analysis:

Videos were analysed and J- and C-like patterns were detected.

__

# Result  <a name="data-overview"></a>

The device and methodology were successfully used to extract tail pattern information. The presented technique can be applied to understand the biological and genetic pathways involved in zebrafish response to electric stimulus and a more comprehensive range of stimuli, such as chemical and mechanical cues.  

__

# Further Information  <a name="data-overview"></a>

Presented Work at 25th International Conference on Miniaturized Systems for Chemistry and Life Sciences (µTAS 2021): [Electrically Induced Movement Patterns in Zebrafish Using Microfluidics and Computer-Aided Analysis]

