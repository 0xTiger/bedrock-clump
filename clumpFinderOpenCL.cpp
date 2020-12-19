#include <iostream>
#include <CL/cl.hpp>
#include <fstream>
#include <vector>
#include <math.h>
#include "PrecomputedRandAdvance.h"
#include <chrono>
#include <tuple>
#include <iomanip>

typedef std::chrono::high_resolution_clock Clock;

#define MAXPASS 10

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

std::vector<int> spiral(int n) {
	n++;
	int k = ceil((sqrt(n) - 1) / 2);
	int t = 2 * k + 1;
	int m = t*t;
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
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	_ASSERT(platforms.size() > 0);

	auto platform = platforms.front();

	std::vector<cl::Device> devices;

	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	auto& device = devices.front();
	auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
	auto version = device.getInfo<CL_DEVICE_VERSION>();
	std::string name = device.getInfo<CL_DEVICE_NAME>();

	std::cout << devices.size() << " devices found" << std::endl;
	std::cout << vendor << ' ' << name << std::endl << version << std::endl;

	std::ifstream File("ker.cl"); //load the kernels from file -> sources -> program
	std::string src(std::istreambuf_iterator<char>(File), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	cl::Context context(devices);
	cl::Program program(context, sources);
	cl::CommandQueue queue(context, device);

	auto err = program.build("-cl-std=CL1.2");

	std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
	std::cerr << "Build log for " << name << ":" << std::endl
		<< buildlog << std::endl; //Determine if there are kernel errors

	if (argc != 3) {
		std::cout << "Please enter integer arguments of the form <start> <end> to specify" << std::endl;
		std::cout << "the range to be searched. Both should be >0 and <67,000,000" << std::endl;
		return 0;
	}
	const int len = 8192;
	const size_t start = atoi(argv[1]); // 0;
	const size_t end = atoi(argv[2]); // start + 15;


	std::vector<unsigned char> bedrock(len * len, 0);
	std::vector<int> offset = { 0, 0 };
	std::tuple<size_t, int, int> best = { 0, 0, 0 };


	cl::Kernel kernel_bedrock(program, "getBedrockTile", &err);
	cl::Kernel kernel_prepare(program, "labelxPreprocess_int_int", &err);
	cl::Kernel kernel_propagate(program, "label4xMain_int_int", &err);
	cl::Kernel kernel_count(program, "getFrequency", &err);
	cl::Kernel kernel_reduce(program, "reduction", &err);

	cl::Buffer a_buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int64_t) * A_OW_112.size(), A_OW_112.data(), &err);
	cl::Buffer b_buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int64_t) * B_OW_112.size(), B_OW_112.data(), &err);
	cl::Buffer bedrock_buf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, sizeof(unsigned char) * bedrock.size(), &err);

	kernel_bedrock.setArg(0, a_buf);
	kernel_bedrock.setArg(1, b_buf);
	kernel_bedrock.setArg(3, bedrock_buf);
	
	int workGroupSize = kernel_reduce.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
	int numWorkGroups = (len * len) / workGroupSize;
	//std::cout << numWorkGroups << std::endl;
	std::vector<int> label(len * len, 0);
	std::vector<int> flags(MAXPASS + 1, 0);
	std::vector<int> freq(len * len, 0);
	std::vector<int> final(numWorkGroups, 0);
	std::vector<int> finalIds(numWorkGroups, 0);

	cl::Buffer label_buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, len * len * sizeof(cl_int), label.data());
	cl::Buffer flags_buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (MAXPASS + 1) * sizeof(cl_int), flags.data());
	cl::Buffer freq_buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, len * len * sizeof(cl_int), freq.data());
	cl::Buffer final_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, numWorkGroups * sizeof(int), final.data());
	cl::Buffer finalIds_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, numWorkGroups * sizeof(int), finalIds.data());


	//1000 (*1000*1000) takes 7500ms before new shiny kernel
	//1000 (*1000*1000) takes 2500ms with new shinyish kernel
	//60 (*4096*4096) takes 2000ms with new shinyish kernel
	//15 (*8192*8192) takes 2100ms with new shinyish kernel
	//15 (*8192*8192) takes 2500ms with finished? kernel
	//15 (*8192*8192) takes 1900ms after neglecting to read freq_buf
	// ^ at this point after scaling up to 150 iters, we can do about 10^9 blocks per second

	for (int i = start; i < end; i++) {
		auto t1 = Clock::now();

		offset = { spiral(i)[0]*len , spiral(i)[1]*len};
		//std::cout << offset[0] << ' ' << offset[1] << std::endl;

		cl::Buffer off_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * offset.size(), offset.data(), &err);

		kernel_bedrock.setArg(2, off_buf);
		
		err = queue.enqueueNDRangeKernel(kernel_bedrock, cl::NullRange, cl::NDRange(len, len));
		//err = queue.enqueueReadBuffer(bedrock_buf, CL_FALSE, 0, sizeof(unsigned char) * bedrock.size(), bedrock.data());

		cl::finish();

		kernel_prepare.setArg(0, label_buf);
		kernel_prepare.setArg(1, bedrock_buf);
		kernel_prepare.setArg(2, flags_buf);
		kernel_prepare.setArg(3, MAXPASS);
		kernel_prepare.setArg(4, 0);
		kernel_prepare.setArg(5, len);
		kernel_prepare.setArg(6, len);

		err = queue.enqueueNDRangeKernel(kernel_prepare, cl::NullRange, cl::NDRange(len, len));

		kernel_propagate.setArg(0, label_buf);
		kernel_propagate.setArg(1, bedrock_buf);
		kernel_propagate.setArg(2, flags_buf);
		kernel_propagate.setArg(4, len);
		kernel_propagate.setArg(5, len);
		for (int i = 1; i <= MAXPASS; i++) {

			kernel_propagate.setArg(3, i);
			
			err = queue.enqueueNDRangeKernel(kernel_propagate, cl::NullRange, cl::NDRange(len, len));
		}
		cl::finish();

		
		kernel_count.setArg(0, label_buf);
		kernel_count.setArg(1, freq_buf);

		err = queue.enqueueFillBuffer(freq_buf, 0, 0, len * len * sizeof(cl_int));
		err = queue.enqueueNDRangeKernel(kernel_count, cl::NullRange, cl::NDRange(len, len));
		//err = queue.enqueueReadBuffer(freq_buf, CL_TRUE, 0, len * len * sizeof(cl_int), freq.data());

		kernel_reduce.setArg(0, freq_buf);
		kernel_reduce.setArg(1, sizeof(int)* workGroupSize, nullptr);
		kernel_reduce.setArg(2, final_buf);
		kernel_reduce.setArg(3, finalIds_buf);

		err = queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(len * len), cl::NDRange(workGroupSize));
		err = queue.enqueueReadBuffer(final_buf, CL_TRUE, 0, numWorkGroups * sizeof(int), final.data());
		err = queue.enqueueReadBuffer(finalIds_buf, CL_TRUE, 0, numWorkGroups * sizeof(int), finalIds.data());

		//err = queue.enqueueReadBuffer(label_buf, CL_FALSE, 0, len * len * sizeof(cl_int), label.data());
		//err = queue.enqueueReadBuffer(flags_buf, CL_TRUE, 0, (MAXPASS + 1) * sizeof(cl_int), flags.data());
		cl::finish();

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

		std::cout << i << ' ';
		std::cout << ' ' << record << " @ (" << recordX + offset[0] << ", " << recordZ + offset[1] << ')' << "                             " << std::endl;

		std::chrono::microseconds ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
		int per_sec = (float)(1000000) / ms.count();

		std::cout << per_sec << "tiles/s" << " ETA: " << ms * (end - i) << '\r';

		std::tuple<size_t, int, int> result = {record, recordX + offset[0], recordZ + offset[1] };

		if (std::get<0>(result) > std::get<0>(best)){
			best = result;
		}
	}

	std::cout << "Best found: " << "                             " << std::endl;
	std::cout << std::get<0>(best) << " @ (" << std::get<1>(best) << ", " << std::get<2>(best) << ')' << std::endl;

	
	std::ofstream outfile;
	outfile.open("recordFile.txt", std::ios_base::app); // append instead of overwrite

	outfile << "Searched: " << start << '-' << end << " Best found: " << std::get<0>(best) << " @ (" << std::get<1>(best) << ", " << std::get<2>(best) << ')' << std::endl;
	outfile.close();
}