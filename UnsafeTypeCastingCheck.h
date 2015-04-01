#pragma once 

#include <assert.h>
#include <map> 
#include "llvm/Pass.h" 
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h" 
#include "llvm/IR/Instructions.h" 
#include "llvm/IR/Constants.h" 
#include "llvm/Support/raw_ostream.h" 
#include "llvm/ADT/APInt.h" 
#include "llvm/ADT/APFloat.h" 

using namespace std; 
using namespace llvm; 

namespace utcc {
  bool UTCC_VERBOSE = true; 
  bool UTCC_WARNING_UNHANDLED_INST = true; 
  bool UTCC_WARNING_UNHANDLED_TYPE = false; 

  enum UTCC_TYPE {
    UINT_UT, // unsigned int 
    NINT_UT, // non-negative int 
    INT_UT,  // int
    NFP_UT,  // non-negaitve float 
    FP_UT,   // float 
    UH_UT    // unhandled type 
  }; 

  // ---- some subroutines ---- 
  void utccPrint (string mess) { errs() << mess; }
  void utccEndl () { errs() << "\n"; }
  void utccPrintl (string mess) { utccPrint(mess); utccEndl(); }
  void utccPrintLLVMValue (Value *lv) {
    utccPrint("    => "); 
    lv->print(errs()); 
    utccEndl(); 
  }
  
  void utccWarning (string mess, Value *lv=NULL) {
    mess.insert(0, "[WARNING]: "); 
    utccPrintl(mess); 
    if (lv != NULL) utccPrintLLVMValue(lv); 
  }

  void utccAbort (string mess, Value *lv=NULL) {
    mess.insert(0, "[ERROR]: ");
    utccPrintl(mess); 
    if (lv != NULL) utccPrintLLVMValue(lv); 
    assert(false && "UTCC Abort... Please refere to the above error message..."); 
  }

  string ut2string (UTCC_TYPE ut) {
    switch (ut) {
    case UINT_UT: 
      return string("UINT_UT"); 
      break; 
    case NINT_UT:
      return string("NINT_UT"); 
      break; 
    case INT_UT:
      return string("INT_UT"); 
      break; 
    case NFP_UT: 
      return string("NFP_UT"); 
      break; 
    case FP_UT:
      return string("FP_UT"); 
      break; 
    case UH_UT: 
      return string("UH_UT"); 
      break; 
    default: 
      utccAbort("Unknown UTCC_TYPE..."); 
      return string(""); 
    }
  }

  bool isUTInteger (UTCC_TYPE ut) {
    if (ut == UINT_UT || ut == NINT_UT || ut == INT_UT) return true; 
    else return false; 
  }

  bool isUTFloat (UTCC_TYPE ut) {
    if (ut == NFP_UT || ut == FP_UT) return true; 
    else return false; 
  }

  UTCC_TYPE utSaturate (UTCC_TYPE ut0, UTCC_TYPE ut1) {
    if ((isUTInteger(ut0) && isUTFloat(ut1)) || 
	(isUTFloat(ut0) && isUTInteger(ut1))) 
      utccAbort("Cannot saturate between integer and floating-point types"); 
    if (isUTInteger(ut0) && isUTInteger(ut1)) {
      if (ut0 == UINT_UT || ut1 == UINT_UT) return UINT_UT; 
      if (ut0 == NINT_UT || ut1 == NINT_UT) return NINT_UT; 
      return INT_UT; 
    }
    else if (isUTFloat(ut0) && isUTFloat(ut1)) {
      if (ut0 == NFP_UT || ut1 == NFP_UT) return NFP_UT; 
      return FP_UT; 
    }
    else {
      if (ut0 == UH_UT) return ut1; 
      if (ut1 == UH_UT) return ut0; 
    }
    
    assert(false); 
    // return UH_UT;
  }

  UTCC_TYPE utWiden (UTCC_TYPE ut0, UTCC_TYPE ut1) {
    if ((isUTInteger(ut0) && isUTFloat(ut1)) || 
	(isUTFloat(ut0) && isUTInteger(ut1))) 
      utccAbort("Cannot widen between integer and floating-point types"); 
    if (ut0 == UH_UT || ut1 == UH_UT) return UH_UT; 
    else if (isUTInteger(ut0) && isUTInteger(ut1)) {
      if (ut0 == INT_UT || ut1 == INT_UT) return INT_UT; 
      else if (ut0 == NINT_UT || ut1 == NINT_UT) return NINT_UT; 
      return UINT_UT; 
    }
    else if (isUTFloat(ut0) && isUTFloat(ut1)) {
      if (ut0 == FP_UT || ut1 == FP_UT) return FP_UT; 
      return NFP_UT; 
    }
    else assert(false); 

    assert(false); 
    // return UH_UT; 
  }


  // ====================
  struct UnsafeTypeCastingCheck : public FunctionPass {
    static char ID; 
    UnsafeTypeCastingCheck() : FunctionPass(ID) {}
    map<Value *, UTCC_TYPE> utcc_tmap; // expr -> type 
    map<Value *, UTCC_TYPE> utcc_pmap; // pointer -> pointed type 

    bool code_modified = false; 
    Function *assert_func = NULL; 

    // -- utilities for accessing utcc_tmap and utcc_pmap -- 
    bool isSamePointer (Value *p0, Value *p1); 
    // ---- 
    UTCC_TYPE queryExprType (Value *expr); 
    // ---- 
    bool isVisitedPointer (Value *pt); 
    // ---- 
    UTCC_TYPE queryPointedType (Value *pt); 
    // ---- 
    void setExprType (Value *expr, UTCC_TYPE ut); 
    // ---- 
    void setPointedType (Value *pt, UTCC_TYPE ut); 
    // ----
    UTCC_TYPE llvmT2utccT (Type *itype, Value *expr=NULL); 
    // ----
    void DumpTypeMap (map<Value *, UTCC_TYPE> tmap); 


    // -- handle constant expression -- 
    UTCC_TYPE checkConstantUT (Value *expr); 


    // -- handle binary instruction -- 
    void handleBinaryOperation (Instruction *inst); 

    // -- handle unary instruction -- 
    void handleAllocInstruction (Instruction *inst); 

    // -- handle load instruction -- 
    void handleLoadInstruction (Instruction *inst); 

    // -- handle store instruction -- 
    void handleStoreInstruction (Instruction *inst); 
						
    // -- handle call instruction -- 
    void handleCallInstruction (Instruction *inst); 

    // -- potential unsafe FP2UI type-casting handler -- 
    void potentialUnsafeFP2UIHandler (FPToUIInst *fp2ui, Value *cop); 

    // -- handle FP2UI instruction -- 
    void handleFP2UIInstruction (Function &func, 
				 Instruction *inst); 

    // -- handle FP2SI instruction -- 
    void handleFP2SIInstruction (Instruction *inst); 

    // -- handle UI2FP instruction -- 
    void handleUI2FPInstruction (Instruction *inst); 
    
    // -- handle SI2FP instruction -- 
    void handleSI2FPInstruction (Instruction *inst); 

    // -- handle FPExt instruction -- 
    void handleFPExtInstruction (Instruction *inst); 

    // -- handle FPTrunc instruction -- 
    void handleFPTruncInstruction (Instruction *inst); 

    // -- handle SExt instruction -- 
    void handleSExtInstruction (Instruction *inst); 

    // -- handle trunc instruction -- 
    void handleTruncInstruction (Instruction *inst); 
    
    // -- handle ZExt instruction -- 
    void handleZExtInstruction (Instruction *inst); 

    // -- handle GetElementPtr instruction -- 
    void handleGetElementPtrInstruction (Instruction *inst); 

    // -- handle ExtractValue instruction -- 
    void handleExtractValueInstruction(Instruction *inst); 
    
    // -- handle all instruction -- 
    void handleInstruction (Function &func, 
			    Instruction *inst); 

    // -- handle BitCast instruction -- 
    void handleBitCastInstruction (Instruction *inst); 

    // -- handle select instruction -- 
    void handleSelectInstruction (Instruction *inst); 

    // -- handle phi node -- 
    void handlePHINode (Instruction *inst); 
  

    bool doInitialization(llvm::Module &mod); 

 
    bool runOnFunction(Function &func); 
  }; 
}

