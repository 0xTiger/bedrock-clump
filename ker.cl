// Written by Naoki Shibata shibatch.sf.net@gmail.com 
// http://ito-lab.naist.jp/~n-sibata/cclarticle/index.xhtml

// This program is in public domain. You can use and modify this code for any purpose without any obligation.

// This is an example implementation of a connected component labeling algorithm proposed in the following paper.
// Naoki Shibata, Shinya Yamamoto: GPGPU-Assisted Subpixel Tracking Method for Fiducial Markers,
// Journal of Information Processing, Vol.22(2014), No.1, pp.19-28, 2014-01. DOI:10.2197/ipsjjip.22.19
/*
MIT License

Copyright(c) 2019 coolmann24

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
*/
__kernel void labelxPreprocess_int_int(global int* label, global unsigned char* pix, global int* flags, int maxpass, int bgc, int iw, int ih) {
	const int x = get_global_id(0), y = get_global_id(1);
	const int p0 = y * iw + x;

	if (y == 0 && x < maxpass + 1) {
		flags[x] = x == 0 ? 1 : 0;
	}

	if (x >= iw || y >= ih) return;

	if (pix[p0] == bgc) { label[p0] = -1; return; }
	if (y > 0 && pix[p0] == pix[p0 - iw]) { label[p0] = p0 - iw; return; }
	if (x > 0 && pix[p0] == pix[p0 - 1]) { label[p0] = p0 - 1; return; }
	label[p0] = p0;
}

__kernel void label4xMain_int_int(global int* label, global unsigned char* pix, global int* flags, int pass, int iw, int ih) {
	const int x = get_global_id(0), y = get_global_id(1);
	if (x >= iw || y >= ih) return;
	const int p0 = y * iw + x;

	if (flags[pass - 1] == 0) return;

	int g = label[p0], og = g;

	if (g == -1) return;

	/*for (int yy = -1; yy <= 1; yy++) {
		for (int xx = -1; xx <= 1; xx++) {
			if (0 <= x + xx && x + xx < iw && 0 <= y + yy && y + yy < ih) {
				const int p1 = (y + yy) * iw + x + xx, s = label[p1];
				if (s != -1 && s < g) g = s;
			}
		}
	}*/

	//considers the 4 cardinal neighbours
	int xx = 0, yy = 1;
	if (0 <= x + xx && x + xx < iw && 0 <= y + yy && y + yy < ih) {
		const int p1 = (y + yy) * iw + x + xx, s = label[p1];
		if (s != -1 && s < g) g = s;
	}

	xx = 0, yy = -1;
	if (0 <= x + xx && x + xx < iw && 0 <= y + yy && y + yy < ih) {
		const int p1 = (y + yy) * iw + x + xx, s = label[p1];
		if (s != -1 && s < g) g = s;
	}

	xx = -1, yy = 0;
	if (0 <= x + xx && x + xx < iw && 0 <= y + yy && y + yy < ih) {
		const int p1 = (y + yy) * iw + x + xx, s = label[p1];
		if (s != -1 && s < g) g = s;
	}

	xx = 1, yy = 0;
	if (0 <= x + xx && x + xx < iw && 0 <= y + yy && y + yy < ih) {
		const int p1 = (y + yy) * iw + x + xx, s = label[p1];
		if (s != -1 && s < g) g = s;
	}

	//0,1 1,0 0,-1 -1,0

	for (int j = 0; j < 6; j++) g = label[g];

	if (g != og) {
		atomic_min(&label[og], g);
		atomic_min(&label[p0], g);
		flags[pass] = 1;
	}
}

kernel void getFrequency(global int* labels, global int* freq) {
	int x = get_global_id(0);
	int z = get_global_id(1);

	int label = labels[get_global_size(0) * x + z];
	if (label != -1) {
		atomic_inc(&freq[label]);
	}
		
}

kernel void reduction(global int* inData, local int* localData, global int* outData, global int* outIdData) {
	size_t globalId = get_global_id(0);
	size_t localSize = get_local_size(0);
	size_t localId = get_local_id(0);


	unsigned bits, var = localSize;
	for (bits = 0; var != 0; ++bits) var >>= 1;

	localData[localId] = inData[globalId];
	barrier(CLK_LOCAL_MEM_FENCE);

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
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	

	if (localId == 0) {

		int bitsum = 0; 
		int nextid;
		for (int i = 0; i < bits - 1; i++) {

			nextid = (1 << i) + bitsum;
			bitsum = (localData[nextid] << i) + bitsum;
		}

		int final_id = localData[nextid] ? nextid : nextid - (localSize >> 1);

		outData[get_group_id(0)] = localData[0];
		outIdData[get_group_id(0)] = get_global_id(0) + final_id;
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



inline long rawSeedFromChunk(int x, int z)
{
	return (((long)x * (long)341873128712 + (long)z * (long)132897987541) ^ (long)0x5DEECE66D) & ((((long)1 << 48) - 1));
}


inline int rand5(long raw_seed, long a, long b)
{
	return (int)((((raw_seed * a + b) & (((long)1 << 48) - 1)) >> 17) % ((long)5));
}


inline int precompChunkIndCalcNormal(int x, int y, int z, int nether)
{
	return ((z * 16 + x) * (nether == 1 ? 8 : 4) + ((nether == 1 ? 7 : 3) - y));
}


inline unsigned char getBedrock(int x, int y, int z, global const long* a, global const long* b)
{
	if (y == 0) return 1;
	if (y < 0 || y > 4) return 0;
	int precomp_ind = precompChunkIndCalcNormal(x & 15, y - 1, z & 15, 0);
	return (rand5(rawSeedFromChunk(x >> 4, z >> 4), a[precomp_ind], b[precomp_ind]) >= y)? 1 : 0;
}

kernel void getBedrockTile(global const long* a, global const long* b, global const int* offset, global unsigned char* outData)
{
	int x = get_global_id(0);
	int z = get_global_id(1);

	outData[get_global_size(0) * x + z] = getBedrock(offset[0] + x, 4, offset[1] + z, a, b);
}



