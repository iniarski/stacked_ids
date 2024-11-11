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

