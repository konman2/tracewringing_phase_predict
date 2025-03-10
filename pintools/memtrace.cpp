/*BEGIN_LEGAL 
Intel Open Source License 

Copyright (c) 2002-2016 Intel Corporation. All rights reserved.
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.  Redistributions
in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.  Neither the name of
the Intel Corporation nor the names of its contributors may be used to
endorse or promote products derived from this software without
specific prior written permission.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE INTEL OR
ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
END_LEGAL */
/*! @file
 *  This file contains an ISA-portable PIN tool for tracing memory accesses.
 */

#include "pin.H"
#include "instlib.H"
#include "control_manager.H"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace INSTLIB;
using namespace CONTROLLER;

#if defined(__GNUC__)
#  if defined(__APPLE__)
#    define ALIGN_LOCK __attribute__ ((aligned(16))) /* apple only supports 16B alignment */
#  else
#    define ALIGN_LOCK __attribute__ ((aligned(64)))
#  endif
#else
# define ALIGN_LOCK __declspec(align(64))
#endif

/* ===================================================================== */
/* Global Variables */
/* ===================================================================== */

std::ofstream TraceFile;

ICOUNT icount;

int mem_counter = 0;

CONTROL_MANAGER control("controller_");

BOOL do_trace;

/* ===================================================================== */
/* Commandline Switches */
/* ===================================================================== */

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "pinatrace.out", "specify trace file name");
KNOB<BOOL> KnobValues(KNOB_MODE_WRITEONCE, "pintool",
    "values", "0", "Output memory values reads and written");
KNOB<BOOL>   KnobAddrOnly(KNOB_MODE_WRITEONCE,   "pintool",
   "addr", "0", "only record address");


/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */

static INT32 Usage()
{
    cerr <<
        "This tool produces a memory address trace.\n"
        "For each (dynamic) instruction reading or writing to memory the the ip and ea are recorded\n"
        "\n";

    cerr << KNOB_BASE::StringKnobSummary();

    cerr << endl;

    return -1;
}


static VOID EmitMem(VOID * ea, INT32 size)
{
    if (!KnobValues)
        return;
    
    switch(size)
    {
      case 0:
        TraceFile << setw(1);
        break;
        
      case 1:
        TraceFile << static_cast<UINT32>(*static_cast<UINT8*>(ea));
        break;
        
      case 2:
        TraceFile << *static_cast<UINT16*>(ea);
        break;
        
      case 4:
        TraceFile << *static_cast<UINT32*>(ea);
        break;
        
      case 8:
        TraceFile << *static_cast<UINT64*>(ea);
        break;
        
      default:
        TraceFile.unsetf(ios::showbase);
        TraceFile << setw(1) << "0x";
        for (INT32 i = 0; i < size; i++)
        {
            TraceFile << static_cast<UINT32>(static_cast<UINT8*>(ea)[i]);
        }
        TraceFile.setf(ios::showbase);
        break;
    }
}

static VOID RecordMem(VOID * ip, CHAR r, VOID * addr, INT32 size, BOOL isPrefetch)
{
    if (do_trace)
    {
        ++mem_counter;
        if (KnobAddrOnly) {
            TraceFile << "M " << setw(2+2*sizeof(ADDRINT)) << addr << " 4";
            //TraceFile << setw(2+2*sizeof(ADDRINT)) << addr;
        } else {
            TraceFile << ip << ": " << r << " " << setw(2+2*sizeof(ADDRINT)) << addr << " "
                  << dec << setw(2) << size << " "
                << hex << setw(2+2*sizeof(ADDRINT));
            if (!isPrefetch)
                EmitMem(addr, size);
        }
        TraceFile << endl;
    }
}

static VOID * WriteAddr;
static INT32 WriteSize;

static VOID RecordWriteAddrSize(VOID * addr, INT32 size)
{
    WriteAddr = addr;
    WriteSize = size;
}


static VOID RecordMemWrite(VOID * ip)
{
    RecordMem(ip, 'W', WriteAddr, WriteSize, false);
}

VOID Instruction(INS ins, VOID *v)
{

    // instruments loads using a predicated call, i.e.
    // the call happens iff the load will be actually executed
        
    if (INS_IsMemoryRead(ins) && INS_IsStandardMemop(ins))
    {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordMem,
            IARG_INST_PTR,
            IARG_UINT32, 'R',
            IARG_MEMORYREAD_EA,
            IARG_MEMORYREAD_SIZE,
            IARG_BOOL, INS_IsPrefetch(ins),
            IARG_END);
    }

    if (INS_HasMemoryRead2(ins) && INS_IsStandardMemop(ins))
    {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordMem,
            IARG_INST_PTR,
            IARG_UINT32, 'R',
            IARG_MEMORYREAD2_EA,
            IARG_MEMORYREAD_SIZE,
            IARG_BOOL, INS_IsPrefetch(ins),
            IARG_END);
    }

    // instruments stores using a predicated call, i.e.
    // the call happens iff the store will be actually executed
    if (INS_IsMemoryWrite(ins) && INS_IsStandardMemop(ins))
    {
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordWriteAddrSize,
            IARG_MEMORYWRITE_EA,
            IARG_MEMORYWRITE_SIZE,
            IARG_END);
        
        if (INS_HasFallThrough(ins))
        {
            INS_InsertCall(
                ins, IPOINT_AFTER, (AFUNPTR)RecordMemWrite,
                IARG_INST_PTR,
                IARG_END);
        }
        if (INS_IsBranchOrCall(ins))
        {
            INS_InsertCall(
                ins, IPOINT_TAKEN_BRANCH, (AFUNPTR)RecordMemWrite,
                IARG_INST_PTR,
                IARG_END);
        }
        
    }
}

VOID Handler(EVENT_TYPE ev, VOID *v, CONTEXT *ctxt, VOID *ip, THREADID tid, BOOL bcast)
{
    switch(ev)
    {
	case EVENT_START:
	    std::cout << "START at " << icount.Count() << std::endl;
	    do_trace = true;
	    break;
	case EVENT_STOP:
	    std::cout << "STOP at " << icount.Count() << std::endl;
        std::cout << "Total mem record: " << mem_counter << std::endl;
	    do_trace = false;
        exit(0);
	    break;
	default:
	    ASSERTX(false);
	    break;
    }
}

/* ===================================================================== */

VOID Fini(INT32 code, VOID *v)
{
    //TraceFile << "#eof" << endl;
    
    TraceFile.close();
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char *argv[])
{
    //string trace_header = string("#\n"
    //                             "# Memory Access Trace Generated By Pin\n"
    //                             "#\n");
    do_trace = false;

    if( PIN_Init(argc,argv) )
    {
        return Usage();
    }
    
    icount.Activate();
    control.RegisterHandler(Handler, 0, FALSE);
    control.Activate();

    TraceFile.open(KnobOutputFile.Value().c_str());
    //TraceFile.write(trace_header.c_str(),trace_header.size());
    TraceFile.setf(ios::showbase);
    
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns

    PIN_StartProgram();
    
    RecordMemWrite(0);
    RecordWriteAddrSize(0, 0);
    
    return 0;
}

/* ===================================================================== */
/* eof */
/* ===================================================================== */
