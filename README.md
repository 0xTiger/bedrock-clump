# bedrock-clump
A GPU accelerated program to search the Minecraft world for the largest connected clump of top-level bedrock.
This is implemented in both CUDA & OpenCL and uses code written by @coolmann24 in their [BedrockFinder](https://github.com/coolmann24/BedrockFinderCpp).

## goal
The project has now concluded and the entire y=5 of the Minecraft world has been searched involving 3.6 quadrillion blocks. This took about 300 hours on a modern GPU.
- [x] 0-100K tiles searched
- [x] 100K - 1M tiles searched
- [x] 1M - 10M tiles searched
- [x] 10M - 54M tiles searched

**Pictured below is the largest clump of top-level bedrock in Minecraft 1.12 found at X: 21,783,512, Z: -800,011**
![A clump of 64 connected bedrock](/LargeBedrock64.png)

## downloads
[universal-v0.1.1](https://github.com/0xTiger/bedrock-clump/releases/download/v0.1.1/universal-clumpFinder.zip) *(NVIDIA & AMD)*

[cuda-v0.1.1](https://github.com/0xTiger/bedrock-clump/releases/download/v0.1.1/cuda-clumpFinder.zip) *(NVIDIA cards only) (higher performance)*

## usage
Download the (windows x64 only) binaries from releases or build from source using Cmake.

In command prompt, navigate to the directory containing the executable using `cd Downloads\` etc.

Run the executable using `clumpFinderCUDA.exe <start> <end>`, where `<start>` and `<end>` specify the range to be searched.
 Both should be between 0 and 54,000,000.

The size and coordinates of the largest clump in your scan area are recorded in recordFile.txt
