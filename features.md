### Numeric features

| **Feature** | **Min** | **Max** |
|----------|------------|---------------|
| frame.len | 70 | 3977 |
| radiotap.length | 48 | 64 |
| frame.time_delta | 0.0 | 3.323003 |
| radiotap.dbm_antsignal | -259 | -24 |
| wlan.duration | 0 | 22704 | 


### Categorical features

| **Feature** | **No. Categories** | **Categories** | 
|----------|------------|---------------|
| radiotap.present.tsft | 2 |'0-0-0', '1-0-0'|
| radiotap.channel.freq | 3 |  2417, 2472, 5180 |
| wlan.fc.type | 3 | 0, 1, 2 |
| wlan.fc.subtype | 16 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 |
| wlan.fc.ds | 4 | '0x00000000', '0x00000001', '0x00000002', '0x00000003' |

### Other features
Binary features having values 0 an 1

* radiotap.channel.flags.cck,
* radiotap.channel.flags.ofdm,
* wlan.fc.frag,
* wlan.fc.retry,
* wlan.fc.pwrmgt,
* wlan.fc.moredata,
* wlan.fc.protected,

| File          | Normal Count | Normal (%) | Anomalous Count | Anomalous (%) |
|---------------|--------------|------------|-----------------|---------------|
| Deauth_21     | 49,115       | 98.230%    | 885             | 1.770%        |
| Deauth_22     | 44,799       | 89.598%    | 5,201           | 10.402%       |
| Deauth_23     | 46,384       | 92.768%    | 3,616           | 7.232%        |
| Deauth_24     | 49,091       | 98.182%    | 909             | 1.818%        |
| Deauth_25     | 47,433       | 94.866%    | 2,567           | 5.134%        |
| Deauth_26     | 47,917       | 95.834%    | 2,083           | 4.166%        |
| Deauth_27     | 47,076       | 94.152%    | 2,924           | 5.848%        |
| Deauth_28     | 47,268       | 94.536%    | 2,732           | 5.464%        |
| Deauth_29     | 46,664       | 93.328%    | 3,336           | 6.672%        |
| Deauth_30     | 41,060       | 82.120%    | 8,940           | 17.880%       |
| Deauth_31     | 45,262       | 90.524%    | 4,738           | 9.476%        |
| Deauth_32     | 25,461       | 96.181%    | 1,011           | 3.819%        |
| Disas_28      | 47,881       | 95.762%    | 2,119           | 4.238%        |
| Disas_29      | 47,761       | 95.524%    | 2,238           | 4.476%        |
| Disas_30      | 46,378       | 92.756%    | 3,622           | 7.244%        |
| Disas_31      | 43,840       | 87.680%    | 6,160           | 12.320%       |
| Disas_32      | 49,207       | 98.414%    | 793             | 1.586%        |
| Disas_33      | 43,417       | 86.834%    | 6,583           | 13.166%       |
| Disas_34      | 39,154       | 78.308%    | 10,846          | 21.692%       |
| Disas_35      | 42,800       | 85.600%    | 7,200           | 14.400%       |
| Disas_36      | 41,298       | 82.596%    | 8,702           | 17.404%       |
| Disas_37      | 40,371       | 80.742%    | 9,629           | 19.258%       |
| Disas_38      | 40,742       | 81.484%    | 9,258           | 18.516%       |
| Disas_39      | 42,927       | 85.854%    | 7,073           | 14.146%       |
| Disas_40      | 12,811       | 93.381%    | 908             | 6.619%        |
| RogueAP_24    | 49,972       | 99.944%    | 28              | 0.056%        |
| RogueAP_25    | 49,917       | 99.834%    | 83              | 0.166%        |
| RogueAP_26    | 49,921       | 99.842%    | 79              | 0.158%        |
| RogueAP_27    | 49,929       | 99.858%    | 71              | 0.142%        |
| RogueAP_28    | 49,915       | 99.830%    | 85              | 0.170%        |
| RogueAP_29    | 49,834       | 99.668%    | 166             | 0.332%        |
| RogueAP_30    | 49,943       | 99.886%    | 57              | 0.114%        |
| RogueAP_31    | 49,939       | 99.878%    | 61              | 0.122%        |
| RogueAP_32    | 49,936       | 99.872%    | 64              | 0.128%        |
| RogueAP_33    | 49,930       | 99.864%    | 68              | 0.136%        |
| RogueAP_34    | 49,909       | 99.818%    | 91              | 0.182%        |
| RogueAP_35    | 49,854       | 99.714%    | 143             | 0.286%        |
| RogueAP_36    | 49,931       | 99.866%    | 67              | 0.134%        |
| RogueAP_37    | 49,895       | 99.790%    | 105             | 0.210%        |
| RogueAP_38    | 49,883       | 99.766%    | 117             | 0.234%        |
| RogueAP_39    | 23,169       | 99.897%    | 24              | 0.103%        |
| Krack_25      | 37,688       | 75.376%    | 12,312          | 24.624%       |
| Krack_26      | 33,658       | 67.316%    | 16,342          | 32.684%       |
| Krack_27      | 33,991       | 67.982%    | 16,009          | 32.018%       |
| Krack_28      | 33,166       | 86.161%    | 5,327           | 13.839%       |
| Kr00k_31      | 46,833       | 93.666%    | 3,167           | 6.334%        |
| Kr00k_32      | 48,862       | 97.724%    | 1,138           | 2.276%        |
| Kr00k_33      | 48,559       | 97.118%    | 1,441           | 2.882%        |
| Kr00k_34      | 46,584       | 93.168%    | 3,416           | 6.832%        |
| Kr00k_35      | 44,370       | 100.000%   | 0               | 0.000%        |
| Kr00k_36      | 42,016       | 84.035%    | 7,982           | 15.965%       |
| ...           | ...          | ...        | ...             | ...           |


data/AWID3_preprocessed/Kr00k_37.csv
Label 0: 39574 (79.148%)
Label 3: 10426 (20.852%)
data/AWID3_preprocessed/Kr00k_38.csv
Label 0: 40001 (80.002%)
Label 3: 9999 (19.998%)
data/AWID3_preprocessed/Kr00k_39.csv
Label 0: 39193 (78.386%)
Label 3: 10807 (21.614%)
data/AWID3_preprocessed/Kr00k_40.csv
Label 0: 38575 (77.150%)
Label 3: 11425 (22.850%)
data/AWID3_preprocessed/Kr00k_41.csv
Label 0: 40472 (80.944%)
Label 3: 9528 (19.056%)
data/AWID3_preprocessed/Kr00k_42.csv
Label 0: 40166 (80.332%)
Label 3: 9834 (19.668%)
data/AWID3_preprocessed/Kr00k_43.csv
Label 0: 40587 (81.174%)
Label 3: 9413 (18.826%)
data/AWID3_preprocessed/Kr00k_44.csv
Label 0: 42131 (84.262%)
Label 3: 7869 (15.738%)
data/AWID3_preprocessed/Kr00k_45.csv
Label 0: 45159 (90.318%)
Label 3: 4841 (9.682%)
data/AWID3_preprocessed/Kr00k_46.csv
Label 0: 45977 (91.958%)
Label 3: 4021 (8.042%)
data/AWID3_preprocessed/Kr00k_47.csv
Label 0: 41801 (83.602%)
Label 3: 8199 (16.398%)
data/AWID3_preprocessed/Kr00k_48.csv
Label 0: 41822 (83.644%)
Label 3: 8178 (16.356%)
data/AWID3_preprocessed/Kr00k_49.csv
Label 0: 41746 (83.492%)
Label 3: 8254 (16.508%)
data/AWID3_preprocessed/Kr00k_50.csv
Label 0: 43478 (86.959%)
Label 3: 6520 (13.041%)
data/AWID3_preprocessed/Kr00k_51.csv
Label 0: 39452 (78.904%)
Label 3: 10548 (21.096%)
data/AWID3_preprocessed/Kr00k_52.csv
Label 0: 42771 (85.544%)
Label 3: 7228 (14.456%)
data/AWID3_preprocessed/Kr00k_53.csv
Label 0: 46338 (92.676%)
Label 3: 3662 (7.324%)
data/AWID3_preprocessed/Kr00k_54.csv
Label 0: 39664 (79.328%)
Label 3: 10336 (20.672%)
data/AWID3_preprocessed/Kr00k_55.csv
Label 0: 41233 (82.466%)
Label 3: 8767 (17.534%)
data/AWID3_preprocessed/Kr00k_56.csv
Label 0: 44743 (89.486%)
Label 3: 5257 (10.514%)
data/AWID3_preprocessed/Kr00k_57.csv
Label 0: 46082 (92.166%)
Label 3: 3917 (7.834%)
data/AWID3_preprocessed/Kr00k_58.csv
Label 0: 458 (100.000%)
data/AWID3_preprocessed/Evil_Twin_29.csv
Label 0: 49835 (99.670%)
Label 2: 165 (0.330%)
data/AWID3_preprocessed/Evil_Twin_30.csv
Label 0: 49834 (99.670%)
Label 2: 165 (0.330%)
data/AWID3_preprocessed/Evil_Twin_31.csv
Label 0: 49653 (99.306%)
Label 2: 347 (0.694%)
data/AWID3_preprocessed/Evil_Twin_32.csv
Label 0: 49924 (99.848%)
Label 2: 76 (0.152%)
data/AWID3_preprocessed/Evil_Twin_33.csv
Label 0: 49920 (99.840%)
Label 2: 80 (0.160%)
data/AWID3_preprocessed/Evil_Twin_34.csv
Label 0: 49590 (99.180%)
Label 2: 410 (0.820%)
data/AWID3_preprocessed/Evil_Twin_35.csv
Label 0: 39194 (78.388%)
Label 2: 10806 (21.612%)
data/AWID3_preprocessed/Evil_Twin_36.csv
Label 0: 43416 (86.832%)
Label 2: 6584 (13.168%)
data/AWID3_preprocessed/Evil_Twin_37.csv
Label 0: 45934 (91.868%)
Label 2: 4066 (8.132%)
data/AWID3_preprocessed/Evil_Twin_38.csv
Label 0: 49619 (99.240%)
Label 2: 380 (0.760%)
data/AWID3_preprocessed/Evil_Twin_39.csv
Label 0: 46681 (93.366%)
Label 2: 3317 (6.634%)
data/AWID3_preprocessed/Evil_Twin_40.csv
Label 0: 43857 (87.714%)
Label 2: 6143 (12.286%)
data/AWID3_preprocessed/Evil_Twin_41.csv
Label 0: 41010 (82.020%)
Label 2: 8990 (17.980%)
data/AWID3_preprocessed/Evil_Twin_42.csv
Label 0: 27799 (55.598%)
Label 2: 22201 (44.402%)
data/AWID3_preprocessed/Evil_Twin_43.csv
Label 0: 40121 (80.242%)
Label 2: 9879 (19.758%)
data/AWID3_preprocessed/Evil_Twin_44.csv
Label 0: 49862 (99.724%)
Label 2: 138 (0.276%)
data/AWID3_preprocessed/Evil_Twin_45.csv
Label 0: 49917 (99.834%)
Label 2: 83 (0.166%)
data/AWID3_preprocessed/Evil_Twin_46.csv
Label 0: 43996 (87.994%)
Label 2: 6003 (12.006%)
data/AWID3_preprocessed/Evil_Twin_47.csv
Label 0: 47706 (95.412%)
Label 2: 2294 (4.588%)
data/AWID3_preprocessed/Evil_Twin_48.csv
Label 0: 49884 (99.772%)
Label 2: 114 (0.228%)
data/AWID3_preprocessed/Evil_Twin_49.csv
Label 0: 49773 (99.546%)
Label 2: 227 (0.454%)
data/AWID3_preprocessed/Evil_Twin_50.csv
Label 0: 40146 (80.292%)
Label 2: 9854 (19.708%)
data/AWID3_preprocessed/Evil_Twin_51.csv
Label 0: 49868 (99.736%)
Label 2: 132 (0.264%)
data/AWID3_preprocessed/Evil_Twin_52.csv
Label 0: 49917 (99.834%)
Label 2: 83 (0.166%)
data/AWID3_preprocessed/Evil_Twin_53.csv
Label 0: 40074 (80.148%)
Label 2: 9926 (19.852%)
data/AWID3_preprocessed/Evil_Twin_54.csv
Label 0: 49878 (99.756%)
Label 2: 122 (0.244%)
data/AWID3_preprocessed/Evil_Twin_55.csv
Label 0: 49868 (99.736%)
Label 2: 132 (0.264%)
data/AWID3_preprocessed/Evil_Twin_56.csv
Label 0: 49849 (99.710%)
Label 2: 145 (0.290%)
data/AWID3_preprocessed/Evil_Twin_57.csv
Label 0: 49963 (99.932%)
Label 2: 34 (0.068%)
data/AWID3_preprocessed/Evil_Twin_58.csv
Label 0: 49997 (99.996%)
Label 2: 2 (0.004%)
data/AWID3_preprocessed/Evil_Twin_59.csv
Label 0: 49889 (99.780%)
Label 2: 110 (0.220%)
data/AWID3_preprocessed/Evil_Twin_60.csv
Label 0: 49836 (99.672%)
Label 2: 164 (0.328%)
data/AWID3_preprocessed/Evil_Twin_61.csv
Label 0: 49906 (99.816%)
Label 2: 92 (0.184%)
data/AWID3_preprocessed/Evil_Twin_62.csv
Label 0: 49956 (99.912%)
Label 2: 44 (0.088%)
data/AWID3_preprocessed/Evil_Twin_63.csv
Label 0: 49921 (99.842%)
Label 2: 79 (0.158%)
data/AWID3_preprocessed/Evil_Twin_64.csv
Label 0: 49937 (99.876%)
Label 2: 62 (0.124%)
data/AWID3_preprocessed/Evil_Twin_65.csv
Label 0: 49950 (99.902%)
Label 2: 49 (0.098%)
data/AWID3_preprocessed/Evil_Twin_66.csv
Label 0: 49890 (99.784%)
Label 2: 108 (0.216%)
data/AWID3_preprocessed/Evil_Twin_67.csv
Label 0: 49836 (99.672%)
Label 2: 164 (0.328%)
data/AWID3_preprocessed/Evil_Twin_68.csv
Label 0: 49936 (99.874%)
Label 2: 63 (0.126%)
data/AWID3_preprocessed/Evil_Twin_69.csv
Label 0: 49894 (99.788%)
Label 2: 106 (0.212%)
data/AWID3_preprocessed/Evil_Twin_70.csv
Label 0: 49841 (99.682%)
Label 2: 159 (0.318%)
data/AWID3_preprocessed/Evil_Twin_71.csv
Label 0: 49803 (99.608%)
Label 2: 196 (0.392%)
data/AWID3_preprocessed/Evil_Twin_72.csv
Label 0: 49904 (99.808%)
Label 2: 96 (0.192%)
data/AWID3_preprocessed/Evil_Twin_73.csv
Label 0: 49897 (99.798%)
Label 2: 101 (0.202%)
data/AWID3_preprocessed/Evil_Twin_74.csv
Label 0: 49901 (99.802%)
Label 2: 99 (0.198%)
data/AWID3_preprocessed/Evil_Twin_75.csv
Label 0: 28677 (99.822%)
Label 2: 51 (0.178%)
data/AWID3_preprocessed/(Re)Assoc_22.csv
Label 0: 49972 (99.944%)
Label 1: 28 (0.056%)
data/AWID3_preprocessed/(Re)Assoc_23.csv
Label 0: 49617 (99.240%)
Label 1: 380 (0.760%)
data/AWID3_preprocessed/(Re)Assoc_24.csv
Label 0: 49241 (98.482%)
Label 1: 759 (1.518%)
data/AWID3_preprocessed/(Re)Assoc_25.csv
Label 0: 49499 (98.998%)
Label 1: 501 (1.002%)
data/AWID3_preprocessed/(Re)Assoc_26.csv
Label 0: 49648 (99.298%)
Label 1: 351 (0.702%)
data/AWID3_preprocessed/(Re)Assoc_27.csv
Label 0: 49630 (99.260%)
Label 1: 370 (0.740%)
data/AWID3_preprocessed/(Re)Assoc_28.csv
Label 0: 49561 (99.122%)
Label 1: 439 (0.878%)
data/AWID3_preprocessed/(Re)Assoc_29.csv
Label 0: 49451 (98.906%)
Label 1: 547 (1.094%)
data/AWID3_preprocessed/(Re)Assoc_30.csv
Label 0: 49672 (99.344%)
Label 1: 328 (0.656%)
data/AWID3_preprocessed/(Re)Assoc_31.csv
Label 0: 49647 (99.294%)
Label 1: 353 (0.706%)
data/AWID3_preprocessed/(Re)Assoc_32.csv
Label 0: 49780 (99.560%)
Label 1: 220 (0.440%)
data/AWID3_preprocessed/(Re)Assoc_33.csv
Label 0: 49646 (99.292%)
Label 1: 354 (0.708%)
data/AWID3_preprocessed/(Re)Assoc_34.csv
Label 0: 49611 (99.222%)
Label 1: 389 (0.778%)
data/AWID3_preprocessed/(Re)Assoc_35.csv
Label 0: 49692 (99.384%)
Label 1: 308 (0.616%)
data/AWID3_preprocessed/(Re)Assoc_36.csv
Label 0: 43764 (99.602%)
Label 1: 175 (0.398%)

Total count of labels across all files (by label):
Label 0: 6107634 (92.972%)
Label 1: 119575 (1.820%)
Label 2: 105950 (1.613%)
Label 3: 236163 (3.595%)

[<tf.Tensor: shape=(), dtype=int32, numpy=2537973>, <tf.Tensor: shape=(), dtype=int32, numpy=527883>]

total 3 065 856
attack 17,2%
normal 82,7%