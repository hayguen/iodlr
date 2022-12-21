// Copyright (C) 2018 Intel Corporation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
// OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS IN THE SOFTWARE.
//
// SPDX-License-Identifier: MIT
//
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#if defined(__aarch64__) || defined(__arm__)
# include "../sse2neon/sse2neon.h"
#else
# include <x86intrin.h>
#endif

#include <climits>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>

#include <unistd.h>


#ifndef MAP_HUGETLB
# define MAP_HUGETLB 0x40000
# define NO_DFLT_MAP_HUGETLB 1
#endif

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
# define NO_DFLT_MAP_HUGE_SHIFT 1
#endif

#ifndef MAP_HUGE_64KB
# define MAP_HUGE_64KB (16UL << MAP_HUGE_SHIFT)
# define NO_DFLT_MAP_HUGE_64KB 1
#endif

#ifndef MAP_HUGE_2MB
# define MAP_HUGE_2MB (21UL << MAP_HUGE_SHIFT)
# define NO_DFLT_MAP_HUGE_2MB 1
#endif

#ifndef MAP_HUGE_32MB
# define MAP_HUGE_32MB (25UL << MAP_HUGE_SHIFT)
# define NO_DFLT_MAP_HUGE_32MB 1
#endif

#ifndef MAP_HUGE_1GB
# define MAP_HUGE_1GB (30UL << MAP_HUGE_SHIFT)
# define NO_DFLT_MAP_HUGE_1GB 1
#endif

# define FLAGS_DFLT  ( MAP_ANONYMOUS | MAP_PRIVATE )
# define FLAGS_HDFLT ( MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB )
# define FLAGS_64K   ( MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_64KB )
# define FLAGS_2M    ( MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_2MB )
# define FLAGS_32M   ( MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_32MB )
# define FLAGS_1G    ( MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_1GB )

#define SZ_KB  (1024UL)
#define SZ_MB  (1024UL * 1024UL)


// default page size: 'getconf PAGE_SIZE'  or  'getconf PAGESIZE'
// huge page size:    'grep Hugepagesize: /proc/meminfo'
// preparation of huge pages as root  is or might be required  beforehand:
//   echo "20" |sudo tee /proc/sys/vm/nr_hugepages
// or
//   sudo sysctl -w vm.nr_hugepages=20
//
// and check with
//   cat /proc/sys/vm/nr_hugepages

#if defined(__aarch64__) || defined(__arm__)
# define DFLT_SIZE (512UL * SZ_MB)
  // raspberry pi and similar devices have much less memory
  // Ubuntu 64bit with 'ls /sys/kernel/mm/hugepages' shows 64kB, 2MB, 32MB and 1GB
#else
// x86/64 shows 2MB and 1GB
# define DFLT_SIZE (8192UL * SZ_MB)
#endif



size_t iodlr_get_default_page_size() {
return (size_t)(sysconf(_SC_PAGESIZE));
}

using std::string;
using std::ifstream;
using std::istringstream;
using std::cout;
using std::cerr;
using std::getline;

uint64_t iodlr_procmeminfo(string key) {
    ifstream ifs("/proc/meminfo");
    string map_line;

    while(getline(ifs,map_line)){
        istringstream iss(map_line);
        string keyname;
        uint64_t value;
        string kb;
        iss >> keyname;
        iss >> value;
        iss >> kb;
        if (keyname == key) {
                return value;
        }
    }
    return -1; // nothing found
}

bool iodlr_hp_enabled() {
    uint64_t val = iodlr_procmeminfo("HugePages_Total:");
    if (val > 0 )
        return true;
    else 
        return false;
}

size_t iodlr_get_hp_size() {
    uint64_t val = iodlr_procmeminfo("Hugepagesize:");
    if (val > 0) {
        return val * 1024;
    }
    return -1;
}

static inline unsigned pretty_size_value(size_t N) {
    return (unsigned)(N >= SZ_MB ? (N / SZ_MB) : (N / SZ_KB));
}

static inline const char * pretty_size_unit(size_t N) {
    return (N >= SZ_MB ? "MB" : "kB");
}

void * iodlr_allocate(size_t s, size_t pgsz) {
    int flags=FLAGS_DFLT;
    void *data;
    if (pgsz == 1024UL*SZ_MB)
       flags = FLAGS_1G;
    else if (pgsz == 32UL*SZ_MB)
       flags = FLAGS_32M;
    else if (pgsz == 2UL*SZ_MB)
       flags = FLAGS_2M;
    else if (pgsz == 64UL*SZ_KB)
       flags = FLAGS_64K;

    data = mmap(NULL, s,
                         PROT_READ | PROT_WRITE, flags,
                         -1, 0);
    if (data == MAP_FAILED) {
#if 0
        cerr << "Error: failed to allocate " << pretty_size_value(s) << pretty_size_unit(s)
             << " with page size " << pretty_size_value(pgsz) << pretty_size_unit(pgsz) << "\n";
#endif
        return 0;
    }
    return data;
}

void iodlr_deallocate(char *d, size_t s) {
    int i;
    i = munmap (d, s);
    assert (i != -1);
}

void zerofill(char *d, size_t s) {
    memset (d, 0, s);
}

/// touches (and sums to prevent optimizer throwing away code) index in all strides
char touch(char *d, unsigned nStrides, size_t stride_sz, size_t index, size_t total_sz) {
    unsigned i;
    char a = 0;
    assert (stride_sz < total_sz);

    for (i=0; i < nStrides; i++) {
        assert (i*stride_sz + index < total_sz);
        a += d[i*stride_sz+index];
    }
    return a;
}

int64_t dotest(size_t total_sz, unsigned nStrides, const char *desc, size_t pgsz, size_t dflt_pgsz, size_t dflt_huge_pgsz) {
    size_t i;
    size_t stride = total_sz / nStrides;
    uint64_t start, end;
    cout << "testing " << desc << " pagesize " << pretty_size_value(pgsz) << pretty_size_unit(pgsz) << "..\n";
    start = _rdtsc();
    char *data = (char *)iodlr_allocate(total_sz, pgsz);
    if (!data)
        return -1;
    zerofill(data, total_sz);
    char sa = 0;
    // touch every byte in all strides
    for (i=0; i < dflt_pgsz; i++) {
        sa += touch(data, nStrides, stride, i, total_sz);
    }
    iodlr_deallocate(data, total_sz);
    end = _rdtsc();
    cout << "Cycles for " << pretty_size_value(pgsz) << pretty_size_unit(pgsz) << " = " << (double)(end - start) << "  char sum " << (int)sa << "\n";
    return (end - start);
}

int main (int argc, char **argv)
{
    int64_t hpt = 1, dt = 1;
    size_t total_sz = DFLT_SIZE;
    int testno = 0;
    int verbose = 0;

    if ( 1 < argc && !strcmp("-v", argv[1]))
        verbose = 1;
    if ( verbose +1 < argc )
        total_sz = SZ_MB * atoi(argv[verbose +1]);
    if ( verbose +2 < argc )
        testno = atoi(argv[verbose +2]);

#ifdef NO_DFLT_MAP_HUGETLB
    cerr << "warning: MAP_HUGETLB wasn't defined\n";
#endif
#ifdef NO_DFLT_MAP_HUGE_SHIFT
    cerr << "warning: MAP_HUGE_SHIFT wasn't defined\n";
#endif
#ifdef NO_DFLT_MAP_HUGE_64KB
    cerr << "warning: MAP_HUGE_64KB wasn't defined\n";
#endif
#ifdef NO_DFLT_MAP_HUGE_2MB
    cerr << "warning: MAP_HUGE_2MB wasn't defined\n";
#endif
#ifdef NO_DFLT_MAP_HUGE_32MB
    cerr << "warning: MAP_HUGE_32MB wasn't defined\n";
#endif
#ifdef NO_DFLT_MAP_HUGE_1GB
    cerr << "warning: MAP_HUGE_1GB wasn't defined\n";
#endif

    const bool has_huge_pages = iodlr_hp_enabled();
    const size_t dflt_page_sz = iodlr_get_default_page_size();
    const size_t huge_page_sz = iodlr_get_hp_size();
    const unsigned nStrides = (unsigned)(total_sz / dflt_page_sz);
    size_t nHugePages = (total_sz + huge_page_sz -1) / huge_page_sz;
    bool printHint = (!has_huge_pages || verbose);
    cout << "has huge page support: " << (has_huge_pages ? "on" : "off") << "\n";
    cout << "huge    page size: " << pretty_size_value(huge_page_sz) << pretty_size_unit(huge_page_sz) << "\n";
    cout << "default page size: " << pretty_size_value(dflt_page_sz) << pretty_size_unit(dflt_page_sz) << "\n";

    dt = 1;
    if (!testno || testno == 1) {
        dt  = dotest(total_sz, nStrides, "default ", dflt_page_sz, dflt_page_sz, huge_page_sz);
        if (dt >= 0)
            cout << "default page size took " << (double)dt << " cycles\n";
        else
            printHint = true;
        if (testno) return 0;
    }

    while (true) {

        size_t pg_sz;

        if (!testno || testno == 2) {
            hpt = dotest(total_sz, nStrides, "default huge", huge_page_sz, dflt_page_sz, huge_page_sz);
            if (hpt >= 0)
                cout << "default huge page took " << (double)hpt << " cycles: speedup = " << ((double)dt / (double)hpt)  << "\n";
            else
                printHint = true;
            if (testno) break;
        }

        pg_sz = 64 * SZ_KB;
        if ((!testno || testno == 3) && dflt_page_sz != pg_sz && huge_page_sz != pg_sz) {
            hpt = dotest(total_sz, nStrides, "specific ", pg_sz, dflt_page_sz, huge_page_sz);
            if (hpt >= 0)
                cout << "huge page size took " << (double)hpt << " cycles: speedup = " << ((double)dt / (double)hpt)  << "\n";
            else
                printHint = true;
            if (testno) break;
        }

        pg_sz = 2 * SZ_MB;
        if ((!testno || testno == 4) && dflt_page_sz != pg_sz && huge_page_sz != pg_sz) {
            hpt = dotest(total_sz, nStrides, "specific ", pg_sz, dflt_page_sz, huge_page_sz);
            if (hpt >= 0)
                cout << "huge page size took " << (double)hpt << " cycles: speedup = " << ((double)dt / (double)hpt)  << "\n";
            else
                printHint = true;
            if (testno) break;
        }

        pg_sz = 32 * SZ_MB;
        if ((!testno || testno == 5) && dflt_page_sz != pg_sz && huge_page_sz != pg_sz) {
            hpt = dotest(total_sz, nStrides, "specific ", pg_sz, dflt_page_sz, huge_page_sz);
            if (hpt >= 0)
                cout << "huge page size took " << (double)hpt << " cycles: speedup = " << ((double)dt / (double)hpt)  << "\n";
            else
                printHint = true;
            if (testno) break;
        }

        pg_sz = 1024 * SZ_MB;
        if ((!testno || testno == 6) && dflt_page_sz != pg_sz && huge_page_sz != pg_sz) {
            hpt = dotest(total_sz, nStrides, "specific ", pg_sz, dflt_page_sz, huge_page_sz);
            if (hpt >= 0)
                cout << "huge page size took " << (double)hpt << " cycles: speedup = " << ((double)dt / (double)hpt)  << "\n";
            else
                printHint = true;
            if (testno) break;
        }

        break;
    }

    if (printHint) {
        cerr << "check/test if huge pages are possible with:\n";
        cerr << "  echo " << nHugePages << " | sudo tee /proc/sys/vm/nr_hugepages\n";
        cerr << "or\n";
        cerr << "  sudo sysctl -w vm.nr_hugepages=" << nHugePages << "\n";
    }

    return 0;
}
