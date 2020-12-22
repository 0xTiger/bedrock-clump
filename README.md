# bedrock-clump
A GPU accelerated program to search the minecraft world for the largest connected clump of layer 5 bedrock.
This is implemented in OpenCL and uses code written by @coolmann24 in their [bedrock finder](https://github.com/coolmann24/BedrockFinderCpp).

## downloads
[universal-v0.1.1](https://github.com/0xTiger/bedrock-clump/releases/download/v0.1.1/universal-clumpFinder.zip) *(NVIDIA & AMD)*

[cuda-v0.1.1](https://github.com/0xTiger/bedrock-clump/releases/download/v0.1.1/cuda-clumpFinder.zip) *(NVIDIA cards only) (higher performance)*

## usage
Download the (windows x64 only) binaries from releases.

In command prompt, navigate to the directory containing the executable using `cd Downloads\` etc.

Run the executable using `LargestBedrockGPU.exe <start> <end>`, where `<start>` and `<end>` are integers between 0 and ~67,000,000.

The size and coordinates of the largest clump in your scan area are recorded in recordFile.txt

## goal
If you would like to help find the largest such bedrock clump then contact my discord at Tiger#8265

Thanks :)


- [x] 0-100K searched
- [x] 100K - 1M searched
- [x] 1M - 10M searched
- [ ] 10M - 67M searched

**The current record is 51 at X: 203817, Z: -429317**
![A clump of 51 connected bedrock](/LargeBedrock51.png)
