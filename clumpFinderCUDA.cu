#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "PrecomputedRandAdvance.h"
#include <fstream>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <tuple>
#include "args_parser.h"
#include <limits>
#include <exception>

typedef std::chrono::high_resolution_clock Clock;

#define THREADSPERBLOCK_X 16
#define THREADSPERBLOCK_Y 16
#define THREADSPERBLOCK_2 256

// Return the root of a tree
__device__ unsigned Find(const int* s_buf, unsigned n) {

	unsigned label = s_buf[n];

	while (label - 1 != n) {
		n = label - 1;
		label = s_buf[n];
	}

	return n;

}

// Links together trees containing a and b
__device__ void Union(int* s_buf, unsigned a, unsigned b) {

	bool done;

	do {

		a = Find(s_buf, a);
		b = Find(s_buf, b);

		if (a < b) {
			int old = atomicMin(s_buf + b, a + 1);
			done = (old == b + 1);
			b = old - 1;
		}
		else if (b < a) {
			int old = atomicMin(s_buf + a, b + 1);
			done = (old == a + 1);
			a = old - 1;
		}
		else {
			done = true;
		}

	} while (!done);

}


// Init phase.
// Labels start at value 1, to differentiate them from background, that has value 0.
__global__ void Init(const int* img, int* labels) {

	unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
	int globalSize = gridDim.x * blockDim.x;

	unsigned img_index = row * globalSize + col;
	unsigned labels_index = row * globalSize + col;



	if (img[img_index]) {

		if (row > 0 && img[img_index - globalSize]) {
			labels[labels_index] = labels_index - globalSize + 1;
		}

		else if (col > 0 && img[img_index - 1]) {
			labels[labels_index] = labels_index;
		}

		else {
			labels[labels_index] = labels_index + 1;
		}
	}

}


// Analysis phase.
__global__ void Analyze(int* labels) {

	unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
	int globalSize = gridDim.x * blockDim.x;

	unsigned labels_index = row * globalSize + col;

	unsigned label = labels[labels_index];

	if (label) {

		unsigned index = labels_index;

		while (label - 1 != index) {
			index = label - 1;
			label = labels[index];
		}

		labels[labels_index] = label;
	}
}

__global__ void Reduce(const int* img, int* labels) {

	unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
	int globalSize = gridDim.x * blockDim.x;

	unsigned img_index = row * globalSize + col;
	unsigned labels_index = row * globalSize + col;


	if (img[img_index]) {

		if (col > 0 && img[img_index - 1]) {
			Union(labels, labels_index, labels_index - 1);
		}
	}
}



__device__ int64_t rawSeedFromChunk(int x, int z)
{
	return (((int64_t)x * (int64_t)341873128712 + (int64_t)z * (int64_t)132897987541) ^ (int64_t)0x5DEECE66D) & ((((int64_t)1 << 48) - 1));
}


__device__ int rand5(int64_t raw_seed, int64_t a, int64_t b)
{
	return (int)((((raw_seed * a + b) & (((int64_t)1 << 48) - 1)) >> 17) % ((int64_t)5));
}


__device__ int precompChunkIndCalcNormal(int x, int y, int z, int nether)
{
	return ((z * 16 + x) * (nether == 1 ? 8 : 4) + ((nether == 1 ? 7 : 3) - y));
}


__device__ int getBedrock(int x, int y, int z, const int64_t* a, const int64_t* b)
{
	if (y == 0) return 1;
	if (y < 0 || y > 4) return 0;
	int precomp_ind = precompChunkIndCalcNormal(x & 15, y - 1, z & 15, 0);
	return (rand5(rawSeedFromChunk(x >> 4, z >> 4), a[precomp_ind], b[precomp_ind]) >= y) ? 1 : 0;
}


__global__ void getBedrockTile(const int64_t* a,  const int64_t* b,  const int* offset, int* outData)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.y * blockDim.y + threadIdx.y;
	int globalSize = gridDim.x * blockDim.x;

	outData[globalSize * x + z] = getBedrock(offset[0] + x, 4, offset[1] + z, a, b);
}

__global__ void getFrequency(int* labels, int* freq) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (labels[x] > 0) {
		atomicAdd(&(freq[labels[x] - 1]), 1);
	}
}

__global__ void reduction(int* inData, int* outData, int* outIdData) {
	size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
	size_t localSize = blockDim.x;
	size_t localId = threadIdx.x;

	__shared__ int localData[THREADSPERBLOCK_2];

	unsigned bits, var = localSize;
	for (bits = 0; var != 0; ++bits) var >>= 1;

	localData[localId] = inData[globalId];
	__syncthreads();

	for (int i = localSize >> 1; i > 0; i >>= 1) {
		if (localId < i) {

			//localData[localId] = max(localData[localId], localData[localId + i]);
			if (localData[localId] > localData[localId + i]) {
				localData[localId + i] = 0; // choose left
			}
			else {
				localData[localId] = localData[localId + i];
				localData[localId + i] = 1; // choose right
			}
		}
		__syncthreads();
	}

	if (localId == 0) {

		int bitsum = 0;
		int nextid;
		for (int i = 0; i < bits - 1; i++) {

			nextid = (1 << i) + bitsum;
			bitsum = (localData[nextid] << i) + bitsum;
		}

		int final_id = localData[nextid] ? nextid : nextid - (localSize >> 1);

		outData[blockIdx.x] = localData[0];
		outIdData[blockIdx.x] = globalId + final_id;
	}


	//8 is 2^3 so we do 3 hops
	//0 0 0 0 0 0 0 0
	//6 | 1 | 0 1 | 0 1 1 1 | -> 1 1 1 -> 8
	//3 | 0 | 1 0 | 0 1 0 0 | -> 0
	//9 | 1 | 0 1 | 1 0 1 0 |
	//7 | 1 | 0 0 | 1 0 1 0 | 0 1 0 0 1 1 1 1 |
	//0 1 2 3 4 5 6 7

	//select 0th bit from column 1 (1) 1 << 0 + 0
	//select 1th bit from column 2 (0) 1 << 1 + 1
	//select 2th bit from column 3 (1) 1 << 2 + 2
	//select 5th bit from column 4 (1) 1 << 3 + 5
	/*int bitsum = 0;
	int nextid;
	for (int i = 0; i < bits - 2; i++) {

		nextid = (1 << i) + bitsum;
		bitsum = (bitsum << 1) + localData[nextid];
	}*/
}

std::ostream& operator<<(std::ostream& os, const std::chrono::microseconds& v) {
	// convert to microseconds
	int64_t us = v.count();

	int h = us / ((int64_t)1000 * (int64_t)1000 * 60 * 60);
	us -= h * ((int64_t)1000 * (int64_t)1000 * 60 * 60);

	int m = us / ((int64_t)1000 * (int64_t)1000 * 60);
	us -= m * ((int64_t)1000 * (int64_t)1000 * 60);

	int s = us / ((int64_t)1000 * (int64_t)1000);
	us -= s * ((int64_t)1000 * (int64_t)1000);

	return os << std::setfill('0') << std::setw(2) << h << ':' << std::setw(2) << m
		<< ':' << std::setw(2) << s;
}

void saveLog(std::string filename, size_t start, size_t end, std::tuple<size_t, int, int> best){
	std::ofstream outfile;
	outfile.open(filename, std::ios_base::app); // append instead of overwrite

	outfile << "Searched: " << start << '-' << end << " Best found: " << std::get<0>(best) << " @ (" << std::get<1>(best) << ", " << std::get<2>(best) << ')' << std::endl;
	outfile.close();
}

std::vector<int> spiral(int n) {
	n++;
	int k = ceil((sqrt(n) - 1) / 2);
	int t = 2 * k + 1;
	int m = t * t;
	t = t - 1;

	if (n >= m - t) {
		return { k - (m - n), -k };
	}
	else { m = m - t; }

	if (n >= m - t) {
		return { -k, -k + (m - n) };
	}
	else { m = m - t; }

	if (n >= m - t) {
		return { -k + (m - n), k };
	}
	else { return { k, k - (m - n - t) }; }
}

int main(int argc, char* argv[])
{
	InputParser args(argc, argv);
	int nDevices;

	cudaGetDeviceCount(&nDevices);

	std::cout << nDevices << " devices found" << std::endl;
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << i << " : " << prop.name;
		std::cout << " Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
	}
	std::cout << "---------------" << std::endl;

	const bool hasflag_q = args.cmdOptionExists("-q");
	const bool hasflag_b = args.cmdOptionExists("-b");
	const std::string flag_b = args.getCmdOption("-b");

	if ((argc < 3) || (hasflag_b && std::string(argv[2]) != "-b" && std::string(argv[2]) != "-q")){
		throw std::invalid_argument("usage: ./clumpFinderCUDA <start> <end | -b batchsize> [-q]");
	}

	const int len = 8192;
	const size_t start = std::stoi(argv[1]);
	const size_t end = hasflag_b ? UINT_MAX : std::stoi(argv[2]);

	const size_t d = hasflag_b ? std::stoi(flag_b) : 0;
	size_t part_start = hasflag_b ? start : 0;
	size_t part_end = hasflag_b ? start + d - (start % d) : 0;

	std::vector<int> offset = { 0, 0};
	std::vector<int> final((len * len) / 256, 0);
	std::vector<int> finalIds((len * len) / 256, 0);


	cudaError_t err;

	int* off_d;
	int* bedrock_d;
	int* freq_d;
	int* final_d;
	int* finalIds_d;
	int* labels_d;

	int64_t* a_d, * b_d;
	err = cudaMalloc(&a_d, sizeof(int64_t) * A_OW_112.size());
	err = cudaMalloc(&b_d, sizeof(int64_t) * B_OW_112.size());
	err = cudaMalloc(&off_d, sizeof(int) * offset.size());
	err = cudaMalloc(&bedrock_d, sizeof(int) * len * len);
	err = cudaMalloc(&labels_d, sizeof(int) * len * len);
	err = cudaMalloc(&final_d, sizeof(int) * final.size());
	err = cudaMalloc(&finalIds_d, sizeof(int) * finalIds.size());
	err = cudaMalloc(&freq_d, sizeof(int) * len * len);


	err = cudaMemcpy(a_d, A_OW_112.data(), sizeof(int64_t) * A_OW_112.size(), cudaMemcpyHostToDevice);
	err = cudaMemcpy(b_d, B_OW_112.data(), sizeof(int64_t) * B_OW_112.size(), cudaMemcpyHostToDevice);

	err = cudaMemset(bedrock_d, 0, sizeof(int) * len * len);

	err = cudaMemcpy(final_d, final.data(), sizeof(int) * final.size(), cudaMemcpyHostToDevice);
	err = cudaMemcpy(finalIds_d, finalIds.data(), sizeof(int) * finalIds.size(), cudaMemcpyHostToDevice);


	std::tuple<size_t, int, int> best = { 0, 0, 0 };
	auto part_best = best;

	//1000 (*1000*1000) takes 7500ms before new shiny kernel
	//1000 (*1000*1000) takes 2500ms with new shinyish kernel
	//60 (*4096*4096) takes 2000ms with new shinyish kernel
	//15 (*8192*8192) takes 2100ms with new shinyish kernel
	//15 (*8192*8192) takes 2500ms with finished? kernel
	//15 (*8192*8192) takes 1900ms after neglecting to read freq_buf
	// ^ at this point after scaling up to 150 iters, we can do about 10^9 blocks per second
	//15 (*8192*8192) takes 6000ms after first unclean cuda trial
	//15 (*8192*8192) takes 1000ms after cleaned up cuda memory copys!
	// ^ at this point after scaling up to 150 iters, we can do about 2.5e9 blocks per second
	dim3 DimGrid(len / THREADSPERBLOCK_X, len / THREADSPERBLOCK_Y);
	dim3 DimBlock(THREADSPERBLOCK_X, THREADSPERBLOCK_Y);

	dim3 DimGrid2((len * len) / THREADSPERBLOCK_2);
	dim3 DimBlock2(THREADSPERBLOCK_2);

	for (int i = start; i < end; i++) {
		auto t1 = Clock::now();

		offset = { spiral(i)[0] * len , spiral(i)[1] * len };
		//std::cout << offset[0] << ' ' << offset[1] << std::endl;


		err = cudaMemcpy(off_d, offset.data(), sizeof(int) * offset.size(), cudaMemcpyHostToDevice);
		getBedrockTile << <DimGrid, DimBlock >> > (a_d, b_d, off_d, bedrock_d);

		//err = cudaMemcpy(bedrock.data(), bedrock_d, sizeof(int) * bedrock.size(), cudaMemcpyDeviceToHost);


		//begin labeling clumps
		err = cudaMemset(labels_d, 0, sizeof(int) * len * len);

		Init << <DimGrid, DimBlock >> > (bedrock_d, labels_d);
		Analyze << <DimGrid, DimBlock >> > (labels_d);
		Reduce << <DimGrid, DimBlock >> > (bedrock_d, labels_d);
		Analyze << <DimGrid, DimBlock >> > (labels_d);
		//finish labeling clumps

		//err = cudaMemcpy(bedrock.data(), bedrock_d, sizeof(int) * bedrock.size(), cudaMemcpyDeviceToHost);
		//err = cudaMemcpy(labels.data(), labels_d, sizeof(int) * labels.size(), cudaMemcpyDeviceToHost);

		err = cudaMemset(freq_d, 0, sizeof(int) * len * len);

		getFrequency << <DimGrid2, DimBlock2 >> > (labels_d, freq_d);

		reduction << <DimGrid2, DimBlock2 >> > (freq_d, final_d, finalIds_d);
		//err = cudaMemcpy(freq.data(), freq_d, sizeof(int) * freq.size(), cudaMemcpyDeviceToHost);
		err = cudaMemcpy(final.data(), final_d, sizeof(int) * final.size(), cudaMemcpyDeviceToHost);
		err = cudaMemcpy(finalIds.data(), finalIds_d, sizeof(int) * finalIds.size(), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		int record = 0, recordi = 0;
		for (int i = 0; i < final.size(); i++) {
			if (final[i] > record) {
				recordi = i;
				record = final[i];
			}
		}
		int recordX = finalIds[recordi] / len;
		int recordZ = finalIds[recordi] % len;

		auto t2 = Clock::now();

		if (!hasflag_q){
		std::cout << i << ' ';
		std::cout << ' ' << record << " @ (" << recordX + offset[0] << ", " << recordZ + offset[1] << ')' << "                             " << std::endl;
		}

		std::chrono::microseconds ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
		int per_sec = (float)(1000000) / ms.count();

		if (hasflag_b){
			std::cout << per_sec << "tiles/s"  << " Tile#: " << i << '\r';
		} else {
			std::cout << per_sec << "tiles/s"  << " ETA: " << ms * (end - i) << " Tile#: " << i << '\r';
		}


		std::tuple<size_t, int, int> result = { record, recordX + offset[0], recordZ + offset[1] };

		if (std::get<0>(result) > std::get<0>(best)) {
			best = result;
		}
		if (hasflag_b){
			if (std::get<0>(result) > std::get<0>(part_best)) {
				part_best = result;
			}

			if (i % d == 0 && i != start){
				saveLog("recordFile.txt", part_start, part_end, part_best);
				part_start = i;
				part_end = i + d;
				part_best = {0, 0, 0};
			}
		}
	}

	std::cout << "Best found: " << "                             " << std::endl;
	std::cout << std::get<0>(best) << " @ (" << std::get<1>(best) << ", " << std::get<2>(best) << ')' << std::endl;

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(off_d);
	cudaFree(bedrock_d);
	cudaFree(freq_d);
	cudaFree(final_d);
	cudaFree(finalIds_d);
	cudaFree(labels_d);

	if (!hasflag_b){
		saveLog("recordFile.txt", start, end, best);
	}
}
