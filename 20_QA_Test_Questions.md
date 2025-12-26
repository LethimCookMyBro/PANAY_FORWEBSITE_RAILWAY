# 20 Q&A Test Questions

Questions sourced from your embedded documents for testing the chatbot.

---

## 🟢 EASY (7 Questions)

### Q1
| Difficulty | 🟢 Easy |
|------------|---------|
| **Question** | What should I do before attaching or detaching devices on the FX3G PLC? |
| **Expected Answer** | Turn off the power to the PLC before attaching or detaching devices to prevent equipment failures or malfunctions. |
| **Source Document** | `Mitsubishi PC MELSEC F FX3G Hardware Manual.pdf` |

---

### Q2
| Difficulty | 🟢 Easy |
|------------|---------|
| **Question** | What software is required for setting and diagnosing a CC-Link Bridge Module? |
| **Expected Answer** | GX Works2 or GX Works3 is required for setting and diagnosing the bridge module. |
| **Source Document** | `Mitsubishi PC CC-Link IE Field Network CC-Link Bridge Module User's Manual.pdf` |

---

### Q3
| Difficulty | 🟢 Easy |
|------------|---------|
| **Question** | What products are in the MELIPC MI5000 series? |
| **Expected Answer** | MELIPC MI5000 series includes the MI5122-VW model. |
| **Source Document** | `Mitsubishi PC Industrial PC CC-Link IE Field Network Basic Reference Manual.pdf` |

---

### Q4
| Difficulty | 🟢 Easy |
|------------|---------|
| **Question** | When do transferred parameters become valid on the FX3 PLC? |
| **Expected Answer** | The transferred parameters become valid when the PLC switches from RUN to STOP. When communication settings are changed, you need to cycle the PLC power. |
| **Source Document** | `Mitsubishi PC MELSEC F FX3 Programming Manual.pdf` |

---

### Q5
| Difficulty | 🟢 Easy |
|------------|---------|
| **Question** | What is error code F800H in the Data Collector? |
| **Expected Answer** | F800H is a warning indicating "Exceeded collection cycle" - the collection process was not completed within the specified collection cycle time. |
| **Source Document** | `Mitsubishi Edge Computing Software iQ Edgecross CC-Link IE Field Network Data Collector User's Manual.pdf` |

---

### Q6
| Difficulty | 🟢 Easy |
|------------|---------|
| **Question** | What is error code F389H in the Data Collector? |
| **Expected Answer** | F389H indicates an internal error in the Data Collector. Corrective actions include restarting the Industrial PC or reinstalling the Data Collector. |
| **Source Document** | `Mitsubishi Edge Computing Software iQ Edgecross CC-Link IE Field Network Data Collector User's Manual.pdf` |

---

### Q7
| Difficulty | 🟢 Easy |
|------------|---------|
| **Question** | Which CPU is used for measuring Data Collector performance? |
| **Expected Answer** | RCPUs are used for measuring performance in the environment where there is no error. |
| **Source Document** | `Mitsubishi Edge Computing Software iQ Edgecross CC-Link IE Field Network Data Collector User's Manual.pdf` |

---

## 🟡 MEDIUM (7 Questions)

### Q8
| Difficulty | 🟡 Medium |
|------------|-----------|
| **Question** | How do I clear the ERR.LED on the CC-Link side after fixing a station number error on the Bridge Module? |
| **Expected Answer** | After correcting the station number setting, turn off then on or reset the power supply of the bridge module to turn off the ERR.LED and clear the data stored in Station number in use status (address: 698H to 69BH). |
| **Source Document** | `Mitsubishi PC CC-Link IE Field Network CC-Link Bridge Module User's Manual.pdf` |

---

### Q9
| Difficulty | 🟡 Medium |
|------------|-----------|
| **Question** | What happens to latched devices of the CPU module when powered off and on or reset? |
| **Expected Answer** | If data in latched devices of the CPU module is cleared to zero in a program when the CPU module is powered off and on or reset, the data may be output without being updated properly. This is a precaution to consider during link refresh. |
| **Source Document** | `Mitsubishi PC Industrial PC CC-Link IE Field Network Basic Reference Manual.pdf` |

---

### Q10
| Difficulty | 🟡 Medium |
|------------|-----------|
| **Question** | How do I access the Basic Settings for MELSEC iQ-R and iQ-L CPU modules? |
| **Expected Answer** | Navigate to: [Navigation window] → [Parameter] → [CPU module model name] → [Module Parameter] → [Basic Settings] |
| **Source Document** | `Mitsubishi PC Industrial PC CC-Link IE Field Network Basic Reference Manual.pdf` |

---

### Q11
| Difficulty | 🟡 Medium |
|------------|-----------|
| **Question** | How many groups can be organized in CC-Link IE Field Network, and which CPUs have no restriction? |
| **Expected Answer** | Up to four groups can be organized. The R00CPU, R01CPU, and R02CPU have no restriction on the group number setting. |
| **Source Document** | `Mitsubishi PC Industrial PC CC-Link IE Field Network Basic Reference Manual.pdf` |

---

### Q12
| Difficulty | 🟡 Medium |
|------------|-----------|
| **Question** | How is link refresh assigned in CC-Link IEF Basic settings? |
| **Expected Answer** | Link refresh is assigned in "Refresh Settings" under "CC-Link IEF Basic Setting". A reserved station is also included in the refresh range. |
| **Source Document** | `Mitsubishi PC Industrial PC CC-Link IE Field Network Basic Reference Manual.pdf` |

---

### Q13
| Difficulty | 🟡 Medium |
|------------|-----------|
| **Question** | What interval should be used for DIN rail mounting screws when installing the Bridge Module? |
| **Expected Answer** | Tighten the screws at intervals of 200mm or less. Use DIN rail types TH35-7.5Fe or TH35-7.5Al. |
| **Source Document** | `Mitsubishi PC CC-Link IE Field Network CC-Link Bridge Module User's Manual.pdf` |

---

### Q14
| Difficulty | 🟡 Medium |
|------------|-----------|
| **Question** | What data types are supported in the Data Collector settings? |
| **Expected Answer** | The Data Collector supports various data types for setting values in its configuration. The setting value data types are specified in the instruction target settings. |
| **Source Document** | `Mitsubishi Edge Computing Software iQ Edgecross CC-Link IE Field Network Data Collector User's Manual.pdf` |

---

## 🔴 HARD (6 Questions)

### Q15
| Difficulty | 🔴 Hard |
|------------|---------|
| **Question** | Explain what happens to cyclic data outputs when the CPU module is in STOP state for different MELSEC series. |
| **Expected Answer** | For MELSEC iQ-R, iQ-L, Q, and L series: Data is held. However, when the device set to perform link refresh is Y device, the data is cleared. |
| **Source Document** | `Mitsubishi PC Industrial PC CC-Link IE Field Network Basic Reference Manual.pdf` |

---

### Q16
| Difficulty | 🔴 Hard |
|------------|---------|
| **Question** | How do you organize groups in CC-Link IE Field Network for remote stations? |
| **Expected Answer** | Organize groups by dividing remote stations into groups. Organizing two or more groups can configure a network with remote stations. Before using the group number setting, check the versions of the CPU module and engineering tool for compatibility. |
| **Source Document** | `Mitsubishi PC Industrial PC CC-Link IE Field Network Basic Reference Manual.pdf` |

---

### Q17
| Difficulty | 🔴 Hard |
|------------|---------|
| **Question** | What European Directive does the FX3G comply with and how was compliance demonstrated? |
| **Expected Answer** | The FX3G products have shown compliance through direct testing (of identified standards) and design analysis (through the creation of a technical construction file) to the European Directive for Low Voltage (2006/95/EC) when used as directed by appropriate documentation. |
| **Source Document** | `Mitsubishi PC MELSEC F FX3G Hardware Manual.pdf` |

---

### Q18
| Difficulty | 🔴 Hard |
|------------|---------|
| **Question** | Explain the relationship between collection cycle and collection processing time in the Data Collector, and what happens when processing time exceeds the cycle. |
| **Expected Answer** | The collection cycle is the interval at which data is collected. Collection processing time is the actual time taken to complete the collection. When processing time exceeds the cycle time, error F800H (Exceeded collection cycle warning) occurs. |
| **Source Document** | `Mitsubishi Edge Computing Software iQ Edgecross CC-Link IE Field Network Data Collector User's Manual.pdf` |

---

### Q19
| Difficulty | 🔴 Hard |
|------------|---------|
| **Question** | What are all the corrective actions for internal errors in the Data Collector? |
| **Expected Answer** | For internal errors in the Data Collector (like F389H), the corrective actions are: 1) Restart the Industrial PC, 2) Reinstall the Data Collector software. |
| **Source Document** | `Mitsubishi Edge Computing Software iQ Edgecross CC-Link IE Field Network Data Collector User's Manual.pdf` |

---

### Q20
| Difficulty | 🔴 Hard |
|------------|---------|
| **Question** | What versions of CPU module and engineering tool are needed for added or enhanced functions in CC-Link IE Field Network? |
| **Expected Answer** | The added or enhanced functions require specific CPU module firmware versions and GX Works3 software versions. Verify compatibility in the documentation before using new features. |
| **Source Document** | `Mitsubishi PC Industrial PC CC-Link IE Field Network Basic Reference Manual.pdf` |

---

## Summary Table

| # | Difficulty | Question Summary | Source Document |
|---|------------|------------------|-----------------|
| 1 | 🟢 Easy | Power off before FX3G attach/detach | FX3G Hardware Manual |
| 2 | 🟢 Easy | Software for Bridge Module | CC-Link Bridge Module Manual |
| 3 | 🟢 Easy | MELIPC MI5000 products | CC-Link IE Field Basic Reference |
| 4 | 🟢 Easy | When parameters become valid | FX3 Programming Manual |
| 5 | 🟢 Easy | F800H error meaning | Data Collector Manual |
| 6 | 🟢 Easy | F389H error meaning | Data Collector Manual |
| 7 | 🟢 Easy | CPU for performance tests | Data Collector Manual |
| 8 | 🟡 Medium | Clear ERR.LED Bridge Module | CC-Link Bridge Module Manual |
| 9 | 🟡 Medium | Latched devices behavior | CC-Link IE Field Basic Reference |
| 10 | 🟡 Medium | Access Basic Settings iQ-R | CC-Link IE Field Basic Reference |
| 11 | 🟡 Medium | Group organization limits | CC-Link IE Field Basic Reference |
| 12 | 🟡 Medium | Link refresh assignment | CC-Link IE Field Basic Reference |
| 13 | 🟡 Medium | DIN rail screw intervals | CC-Link Bridge Module Manual |
| 14 | 🟡 Medium | Data types in Data Collector | Data Collector Manual |
| 15 | 🔴 Hard | Cyclic data in STOP state | CC-Link IE Field Basic Reference |
| 16 | 🔴 Hard | Group organization method | CC-Link IE Field Basic Reference |
| 17 | 🔴 Hard | FX3G EU Directive compliance | FX3G Hardware Manual |
| 18 | 🔴 Hard | Collection cycle vs processing time | Data Collector Manual |
| 19 | 🔴 Hard | Internal error corrective actions | Data Collector Manual |
| 20 | 🔴 Hard | Version requirements enhanced functions | CC-Link IE Field Basic Reference |

---

## Document Sources

All questions are sourced from these 5 embedded documents:

1. **Mitsubishi PC MELSEC F FX3G Hardware Manual.pdf** (3 questions)
2. **Mitsubishi PC CC-Link IE Field Network CC-Link Bridge Module User's Manual.pdf** (3 questions)
3. **Mitsubishi PC MELSEC F FX3 Programming Manual.pdf** (1 question)
4. **Mitsubishi PC Industrial PC CC-Link IE Field Network Basic Reference Manual.pdf** (8 questions)
5. **Mitsubishi Edge Computing Software iQ Edgecross CC-Link IE Field Network Data Collector User's Manual.pdf** (5 questions)
