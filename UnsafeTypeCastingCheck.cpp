#include <assert.h>
#include <map> 
#include "llvm/Pass.h" 
#include "llvm/IR/Function.h" 
#include "llvm/IR/Instructions.h" 
#include "llvm/IR/Constants.h" 
#include "llvm/IR/Type.h" 
#include "llvm/IR/InstrTypes.h" 
#include "llvm/Support/raw_ostream.h" 
#include "llvm/ADT/APInt.h" 
#include "llvm/ADT/APFloat.h" 
#include "llvm/ADT/ArrayRef.h" 
#include "llvm/IR/GlobalAlias.h" 

#include "UnsafeTypeCastingCheck.h" 

using namespace std; 
using namespace llvm; 
using namespace utcc; 


// ====================
bool UnsafeTypeCastingCheck::isSamePointer (Value *p0, Value *p1) {  
  if (p0 == p1) return true; 
  else if (isa<GetElementPtrInst>(p0) && 
	   isa<GetElementPtrInst>(p1)) {
    GetElementPtrInst *g0 = dyn_cast<GetElementPtrInst>(p0); 
    GetElementPtrInst *g1 = dyn_cast<GetElementPtrInst>(p1); 

    assert(g0 != NULL && g1 != NULL); 
    if (g0->getNumOperands() == g1->getNumOperands()) {
      unsigned int n_ops = g0->getNumOperands(); 
      Value *base0 = g0->getOperand(0); 
      Value *base1 = g1->getOperand(0); 

      if (isSamePointer(base0, base1)) {
	unsigned int i; 
	for (i = 1 ; i < n_ops ; i++) {
	  if (g0->getOperand(i) != g1->getOperand(i)) break; 
	}
	if (i == n_ops) return true; 
      }
    }
  }
  else ; 

  return false; 
}
// ---- 
UTCC_TYPE UnsafeTypeCastingCheck::queryExprType (Value *expr) {
  assert(expr != NULL); 
  
  if (utcc_tmap.find(expr) != utcc_tmap.end()) 
    return utcc_tmap[expr]; 
  else {
    if (isa<Constant>(expr)) 
      return checkConstantUT(expr); 
    
    DumpTypeMap(utcc_tmap); 
    utccAbort("This Expression Was Not properly Handled in Prior...", expr); 
  }
}
// ---- 
bool UnsafeTypeCastingCheck::isVisitedPointer (Value *pt) {
  assert(pt != NULL); 
  assert(pt->getType()->isPointerTy()); 

  if (utcc_pmap.find(pt) != utcc_pmap.end()) 
    return true; 

  for (map<Value *, UTCC_TYPE>::iterator mit = utcc_pmap.begin() ; 
       mit != utcc_pmap.end() ; 
       mit++) {
    if (isSamePointer(mit->first, pt)) return true; 
  }  

  return false; 
}
// ---- 
UTCC_TYPE UnsafeTypeCastingCheck::queryPointedType (Value *pt) {
  assert(pt != NULL); 
  assert(pt->getType()->isPointerTy()); 
  
  if (utcc_pmap.find(pt) != utcc_pmap.end())
    return utcc_pmap[pt]; 

  for (map<Value *, UTCC_TYPE>::iterator mit = utcc_pmap.begin() ; 
       mit != utcc_pmap.end() ; 
       mit++) {
    if (isSamePointer(mit->first, pt))
      return mit->second; 
  }
  
  DumpTypeMap(utcc_pmap); 
  utccAbort("This Pointer Was Not properly Handled in Prior...", pt); 
}
// ---- 
void UnsafeTypeCastingCheck::setExprType (Value *expr, UTCC_TYPE ut) {
  assert(expr != NULL); 
  
  if (utcc_tmap.find(expr) != utcc_tmap.end()) 
    utcc_tmap[expr] = ut; 
  else 
    utcc_tmap.insert(pair<Value *, UTCC_TYPE>(expr, ut)); 
}
// ---- 
void UnsafeTypeCastingCheck::setPointedType (Value *pt, UTCC_TYPE ut) {
  assert(pt != NULL); 

  map<Value *, UTCC_TYPE>::iterator mit; 
  for (mit = utcc_pmap.begin() ; 
       mit != utcc_pmap.end() ; 
       mit++) {
    if (isSamePointer(mit->first, pt)) {
      mit->second = ut; 
      break; 
    } 
  }
  if (mit == utcc_pmap.end()) {
    if (utcc_pmap.find(pt) != utcc_pmap.end()) 
      utcc_pmap[pt] = ut; 
    else 
      utcc_pmap.insert(pair<Value *, UTCC_TYPE>(pt, ut)); 
  }
}
// ----
UTCC_TYPE UnsafeTypeCastingCheck::llvmT2utccT (Type *itype, Value *expr) { 
  if(itype->isIntegerTy()) return INT_UT; 
  else if (itype->isFloatingPointTy()) return FP_UT; 
  else {
    if (UTCC_WARNING_UNHANDLED_TYPE) {
      utccWarning("Unhandled LLVM TYPE", expr); 
      utccPrint("    : "); 
      itype->print(errs()); 
      utccPrintl(" -> conservatively handled by casting to Unknown"); 
    }
    return UH_UT; 
  }
}
// ----
void UnsafeTypeCastingCheck::DumpTypeMap (map<Value *, UTCC_TYPE> tmap) {
  errs() << "---- tmap (" << tmap.size() << ") ----\n"; 
  
  map<Value *, UTCC_TYPE>::iterator mit; 
  for (mit = tmap.begin() ; 
       mit != tmap.end() ; 
       mit++) {
    mit->first->print(errs()); 
    utccPrint("  =>  "); 
    utccPrintl(ut2string(mit->second)); 
    utccPrintl("-------------------"); 
  }
}


// -- handle constant expression -- 
UTCC_TYPE UnsafeTypeCastingCheck::checkConstantUT (Value *expr) {
  if (!isa<Constant>(expr))
    utccAbort("checkConstantUT cannot handle a non-constant expression"); 
  
  if (ConstantInt *cint = dyn_cast<ConstantInt>(expr)) {
    if (cint->getValue().isNonNegative()) return NINT_UT; 
    else return INT_UT; 
  }
  else if (ConstantFP *cfp = dyn_cast<ConstantFP>(expr)) {
    if (cfp->getValueAPF().isNegative()) return FP_UT;
    else return NFP_UT; 
  }
  else return UH_UT; 
}


// -- handle binary instruction -- 
void UnsafeTypeCastingCheck::handleBinaryOperation (Instruction *inst) {
  assert(inst->getNumOperands() == 2);       
  
  Value *lhs = inst->getOperand(0); 
  Value *rhs = inst->getOperand(1); 
  
  UTCC_TYPE lht = queryExprType(lhs); 
  UTCC_TYPE rht = queryExprType(rhs); 
  
  switch (inst->getOpcode()) {
  case Instruction::Sub:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::FSub:
    if ((lht == UINT_UT || 
	 lht == NINT_UT || 
	 lht == INT_UT) && 
	(rht == UINT_UT || 
	 rht == NINT_UT || 
	 rht == INT_UT)) {
      setExprType(inst, INT_UT); 
    }
    else if ((lht == NFP_UT || 
	      lht == FP_UT) && 
	     (rht == NFP_UT || 
	      rht == FP_UT)) {
      setExprType(inst, FP_UT); 
    }
    else {
      utccAbort("Unhandled Cases of Bop's Operand Types"); 
    }
    break; 
    
  case Instruction::Add:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FAdd:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    if (lht == rht) {
      setExprType(inst, lht); 
    }
    else if ((lht == NINT_UT && rht == INT_UT) || 
	     (lht == INT_UT && rht == NINT_UT) || 
	     (lht == UINT_UT && rht == INT_UT) || 
	     (lht == INT_UT && rht == UINT_UT)) {
      setExprType(inst, INT_UT); 
    }
    else if ((lht == NINT_UT && rht == UINT_UT) || 
	     (lht == UINT_UT && rht == NINT_UT)) {
      setExprType(inst, NINT_UT); 
    }
    else if ((lht == NFP_UT && rht == FP_UT) || 
	     (lht == FP_UT && rht == NFP_UT)) {
      setExprType(inst, FP_UT); 
    }
    else {
      utccAbort("Unhandled Cases of Bop's Operand Types"); 
    }
    break;
    
  default: 
    utccAbort("This binary operation is not handled by handleBinaryOperation", inst); 
    break; 
  }
}


// -- handle unary instruction -- 
void UnsafeTypeCastingCheck::handleAllocInstruction (Instruction *inst) {
  AllocaInst *ainst = dyn_cast<AllocaInst>(inst); 
  if (ainst == NULL) 
    utccAbort("handleAllocInstruction cannot process with a non-alloca instruction"); 
  Type *atype = ainst->getAllocatedType(); 
  setPointedType(ainst, llvmT2utccT(atype, ainst));
}


// -- handle load instruction -- 
void UnsafeTypeCastingCheck::handleLoadInstruction (Instruction *inst) {
  LoadInst *linst = dyn_cast<LoadInst>(inst); 
  if (linst == NULL) 
    utccAbort("handleLoadInstruction cannot process with a non-load instruction"); 

  Value *pt = linst->getPointerOperand(); 
  
  // check if pointing to global value (NOTE: a global value is always a pointer) 
  if (ConstantExpr *cexpr = dyn_cast<ConstantExpr>(pt)) {
    if (cexpr->getOpcode() == Instruction::GetElementPtr && 
	cexpr->getNumOperands() >= 1) {
      GlobalValue *gpointer = dyn_cast<GlobalValue>(cexpr->getOperand(0)); 
      pt = gpointer; 
      UTCC_TYPE ptt = queryPointedType(pt); 
      setExprType(linst, ptt); 
      return; 
    }
  }

  // the final (default) handle 
  UTCC_TYPE ptt = queryPointedType(pt); 
  setExprType(linst, ptt); 
}


// -- handle store instruction -- 
void UnsafeTypeCastingCheck::handleStoreInstruction (Instruction *inst) {
  StoreInst *sinst = dyn_cast<StoreInst>(inst); 
  if (sinst == NULL) 
    utccAbort("handleStoreInstruction cannot process with a non-store instruction"); 
  Value *pt = sinst->getPointerOperand(); 
  Value *vl = sinst->getValueOperand(); 
  UTCC_TYPE ptt = queryPointedType(pt); 
  UTCC_TYPE vlt = queryExprType(vl); 
  
  setPointedType(pt, vlt); 
}

						
// -- handle call instruction -- 
void UnsafeTypeCastingCheck::handleCallInstruction (Instruction *inst) {
  CallInst *cinst = dyn_cast<CallInst>(inst); 
  if (cinst == NULL) 
    utccAbort("handleCallInstruction cannot process with a non-call instruction");       
  Type *ctype = cinst->getType(); 
  
  string func_name = cinst->getCalledFunction()->getName().str(); 
  
  if (func_name.compare("fabs") == 0 || 
      func_name.compare("sqrt") == 0 || 
      func_name.compare("exp") == 0) {
    setExprType(cinst, NFP_UT); 
  }
  else if (func_name.compare("ceil") == 0 || 
	   func_name.compare("floor") == 0) {
    assert(inst->getNumOperands() == 2); 
    Value *arg = inst->getOperand(0); 
    UTCC_TYPE argt = queryExprType(arg); 
    setExprType(cinst, argt); 
  }
  else if (func_name.compare("max") == 0) {
    assert(inst->getNumOperands() == 3); 
    Value *op0 = inst->getOperand(0); 
    Value *op1 = inst->getOperand(1); 
    UTCC_TYPE t0 = queryExprType(op0); 
    UTCC_TYPE t1 = queryExprType(op1); 

    if (t0 == NFP_UT || t1 == NFP_UT) 
      setExprType(cinst, NFP_UT); 
    else 
      setExprType(cinst, FP_UT); 
  }
  else if (func_name.compare("min") == 0) {
    assert(inst->getNumOperands() == 3); 
    Value *op0 = inst->getOperand(0); 
    Value *op1 = inst->getOperand(1); 
    UTCC_TYPE t0 = queryExprType(op0); 
    UTCC_TYPE t1 = queryExprType(op1); 

    if (t0 == NFP_UT && t1 == NFP_UT) 
      setExprType(cinst, NFP_UT); 
    else 
      setExprType(cinst, FP_UT); 
  }
  else if (func_name.compare("claimNonNegativeInt") == 0 || 
	   func_name.compare("claimNonNegativeUint") == 0) {
    assert(inst->getNumOperands() == 2); 
    Value *arg = inst->getOperand(0); 
    setPointedType(arg, NINT_UT); 
  }
  else if (func_name.compare("claimNonNegativeFP32") == 0 || 
	   func_name.compare("claimNonNegativeFP64") == 0) {
    assert(inst->getNumOperands() == 2); 
    Value *arg = inst->getOperand(0); 
    setPointedType(arg, NFP_UT); 
  }
  else 
    setExprType(cinst, llvmT2utccT(ctype, cinst));
}


// -- potential unsafe FP2UI type-casting handler -- 
void UnsafeTypeCastingCheck::potentialUnsafeFP2UIHandler (FPToUIInst *fp2ui, Value *cop) {
  assert(fp2ui != NULL && 
	 cop != NULL); 
  errs() << "[WARNING]: Potential Unsafe FP2UI Type-casting Detected!! \n"; 
  FCmpInst *fcinst = 
    dyn_cast<FCmpInst>(CmpInst::Create(Instruction::FCmp, 
				       CmpInst::Predicate::FCMP_OGE, 
				       cop, 
				       ConstantFP::get(cop->getType(), 
						       0), 
				       "", 
				       (Instruction *)fp2ui)); 
  assert(fcinst != NULL); 
  SelectInst *sinst = SelectInst::Create(fcinst, 
					 Constant::getIntegerValue(fp2ui->getType(), 
								   APInt(32, 1)), 
					 Constant::getIntegerValue(fp2ui->getType(), 
								   APInt(32, 0)), 
					 "", 
					 (Instruction*)fp2ui); 
  ArrayRef<Value*> assert_args (sinst); 
  CallInst *cinst = CallInst::Create(assert_func, 
				     assert_args, 
				     "", 
				     (Instruction*)fp2ui); 
  code_modified = true; 
}


// -- handle FP2UI instruction -- 
void UnsafeTypeCastingCheck::handleFP2UIInstruction (Function &func, 
						     Instruction *inst) {
  FPToUIInst *fp2ui = dyn_cast<FPToUIInst>(inst); 
  if (fp2ui == NULL) 
    utccAbort("handleFP2UIInstruction cannot process with a non-fp2ui instruction");       
  assert(fp2ui->getNumOperands() == 1); 
  Value *cop = fp2ui->getOperand(0); 
  Type *totype = fp2ui->getType(); 
  setExprType(fp2ui, llvmT2utccT(totype, fp2ui));
  
  UTCC_TYPE fromt = queryExprType(cop); 
  UTCC_TYPE tot = queryExprType(fp2ui); 
  
  if (fromt == NFP_UT) {
    setExprType(fp2ui, UINT_UT); 
  }
  else if (fromt == FP_UT) {
    potentialUnsafeFP2UIHandler(fp2ui, cop); 
  }
  else utccAbort("Invalid from-type of FP2UI casting...", inst); 
}


// -- handle FP2SI instruction -- 
void UnsafeTypeCastingCheck::handleFP2SIInstruction (Instruction *inst) {
  FPToSIInst *fp2si = dyn_cast<FPToSIInst>(inst); 
  if (fp2si == NULL) 
    utccAbort("handleFP2SIInstruction cannot process with a non-fp2si instruction");       
  assert(fp2si->getNumOperands() == 1); 
  Value *cop = fp2si->getOperand(0); 
  Type *totype = fp2si->getType(); 
  setExprType(fp2si, llvmT2utccT(totype, fp2si));
  
  UTCC_TYPE fromt = queryExprType(cop); 
  UTCC_TYPE tot = queryExprType(fp2si); 
  if (fromt == NFP_UT) {
    setExprType(fp2si, NINT_UT); 
  }
  else if (fromt == FP_UT); 
  else utccAbort("Invalid from-type of FP2SI casting...", inst); 
}


// -- handle UI2FP instruction -- 
void UnsafeTypeCastingCheck::handleUI2FPInstruction (Instruction *inst) {
  UIToFPInst *ui2fp = dyn_cast<UIToFPInst>(inst); 
  if (ui2fp == NULL) 
    utccAbort("handleUI2FPInstruction cannot process with a non-ui2fp instruction");       
  assert(ui2fp->getNumOperands() == 1); 
  Value *cop = ui2fp->getOperand(0); 
  Type *totype = ui2fp->getType(); 
  setExprType(ui2fp, llvmT2utccT(totype, ui2fp));
  
  UTCC_TYPE fromt = queryExprType(cop); 
  UTCC_TYPE tot = queryExprType(ui2fp); 
  
  if (fromt == UINT_UT || 
      fromt == NINT_UT) {
    setExprType(ui2fp, NFP_UT); 
  }
  else if (fromt == INT_UT) {
    errs() << "[WARNING]: Potentially Uses a Signed Integer as Unsigned in UI2FP...\n"; 
    setExprType(ui2fp, NFP_UT); 
  }
  else utccAbort("Invalid from-type of UI2FP casting...", inst); 
}

    
// -- handle SI2FP instruction -- 
void UnsafeTypeCastingCheck::handleSI2FPInstruction (Instruction *inst) {
  SIToFPInst *si2fp = dyn_cast<SIToFPInst>(inst); 
  if (si2fp == NULL) 
    utccAbort("handleSI2FPInstruction cannot process with a non-si2fp instruction");       
  assert(si2fp->getNumOperands() == 1); 
  Value *cop = si2fp->getOperand(0); 
  Type *totype = si2fp->getType(); 
  setExprType(si2fp, llvmT2utccT(totype, si2fp));

  UTCC_TYPE fromt = queryExprType(cop); 
  UTCC_TYPE tot = queryExprType(si2fp); 

  if (fromt == UINT_UT || 
      fromt == NINT_UT) {
    setExprType(si2fp, NFP_UT); 
  }
  else if (fromt == INT_UT); 
  else {
    utccAbort("Invalid from-type of SI2FP casting...", inst); 
  }
}


// -- handle FPExt instruction -- 
void UnsafeTypeCastingCheck::handleFPExtInstruction (Instruction *inst) {
  FPExtInst *finst = dyn_cast<FPExtInst>(inst); 
  if (finst == NULL)
    utccAbort("handleFPExtInstruction cannot process with a non-fpext instruction");       
  assert(finst->getNumOperands() == 1); 
  UTCC_TYPE fromt = queryExprType(finst->getOperand(0)); 
  setExprType(finst, fromt); 
}


// -- handle FPTrunc instruction -- 
void UnsafeTypeCastingCheck::handleFPTruncInstruction (Instruction *inst) {
  FPTruncInst *finst = dyn_cast<FPTruncInst>(inst); 
  if (finst == NULL)
    utccAbort("handleFPTruncInstruction cannot process with a non-fptrunc instruction");       
  assert(finst->getNumOperands() == 1); 
  UTCC_TYPE fromt = queryExprType(finst->getOperand(0)); 
  setExprType(finst, fromt); 
}


// -- handle SExt instruction -- 
void UnsafeTypeCastingCheck::handleSExtInstruction (Instruction *inst) {
  SExtInst *finst = dyn_cast<SExtInst>(inst); 
  if (finst == NULL)
    utccAbort("handleSExtInstruction cannot process with a non-sext instruction");       
  assert(finst->getNumOperands() == 1); 
  UTCC_TYPE fromt = queryExprType(finst->getOperand(0)); 
  setExprType(finst, fromt); 
}


// -- handle trunc instruction -- 
void UnsafeTypeCastingCheck::handleTruncInstruction (Instruction *inst) {
  TruncInst *tinst = dyn_cast<TruncInst>(inst); 
  if (tinst == NULL)
    utccAbort("handleTruncInstruction cannot process with a non-trunc instruction");       
  setExprType(tinst, llvmT2utccT(tinst->getType(), tinst)); 
}


// -- handle ZExt instruction -- 
void UnsafeTypeCastingCheck::handleZExtInstruction (Instruction *inst) {
  ZExtInst *finst = dyn_cast<ZExtInst>(inst); 
  if (finst == NULL)
    utccAbort("handleZExtInstruction cannot process with a non-zext instruction");       
  assert(finst->getNumOperands() == 1); 
  UTCC_TYPE fromt = NINT_UT; // queryExprType(finst->getOperand(0)); 
  setExprType(finst, fromt); 
}


// -- handle GetElementPtr instruction -- 
void UnsafeTypeCastingCheck::handleGetElementPtrInstruction (Instruction *inst) {
  GetElementPtrInst * ginst = dyn_cast<GetElementPtrInst>(inst); 
  if (ginst == NULL) 
    utccAbort("handleGetElementPtrInstruction cannot process with a non-getelementptr instruction");       
  Value *pt = ginst->getPointerOperand(); 

  UTCC_TYPE pt_ut_self = UH_UT; 
  UTCC_TYPE pt_ut_base = UH_UT; 
  UTCC_TYPE pt_ut_element = llvmT2utccT(ginst->getType()->getPointerElementType(), ginst); 

  if (isVisitedPointer(ginst)) pt_ut_self = queryPointedType(ginst); 
  if (isVisitedPointer(pt)) pt_ut_base = queryPointedType(pt); 

  setPointedType(ginst, utSaturate(pt_ut_element, 
				   utSaturate(pt_ut_self, pt_ut_base))); 
  setExprType(ginst, llvmT2utccT(ginst->getType(), ginst)); 
}


// -- handle ExtractValue instruction -- 
void UnsafeTypeCastingCheck::handleExtractValueInstruction(Instruction *inst) {
  ExtractValueInst *einst = dyn_cast<ExtractValueInst>(inst); 
  if (einst == NULL) 
    utccAbort("handleExtractValueInstruction cannot process with a non-extractvalue instruction");       
  setExprType(einst, llvmT2utccT(einst->getType())); 
}


// -- handle BitCast instruction -- 
void UnsafeTypeCastingCheck::handleBitCastInstruction (Instruction *inst) {
  BitCastInst *binst = dyn_cast<BitCastInst>(inst); 
  if (binst == NULL) 
    utccAbort("handleBitCastInstruction cannot process with a non-bitcast instruction");       
  Type *fromt = binst->getSrcTy(); 
  Type *tot = binst->getDestTy(); 

  if (tot->isPointerTy()) {
    setExprType(binst, llvmT2utccT(tot, binst)); 
    setPointedType(binst, llvmT2utccT(tot->getPointerElementType(), binst)); 
  }
  else {
    setExprType(binst, llvmT2utccT(tot, binst)); 
  }
}


// -- handle select instruction -- 
void UnsafeTypeCastingCheck::handleSelectInstruction (Instruction *inst) {
  SelectInst *sinst = dyn_cast<SelectInst>(inst); 
  if (sinst == NULL) 
    utccAbort("handleSelectInstruction cannot process with a non-select instruction");       
  assert(sinst->getNumOperands() == 3); 
  Value *choice0 = sinst->getOperand(1); 
  Value *choice1 = sinst->getOperand(2); 
  Type *type0 = choice0->getType();
  Type *type1 = choice1->getType(); 
  UTCC_TYPE ut0 = llvmT2utccT(type0, choice0); 
  UTCC_TYPE ut1 = llvmT2utccT(type1, choice1); 

  if (type0->isIntegerTy() && 
      type1->isIntegerTy()) {
    if (ut0 == ut1) 
      setExprType(sinst, ut0); 
    else if (ut0 == INT_UT || ut1 == INT_UT) 
      setExprType(sinst, INT_UT); 
    else setExprType(sinst, NINT_UT); 
  }
  else if (type0->isFloatingPointTy() && 
	   type1->isFloatingPointTy()) {
    if (ut0 == ut1) 
      setExprType(sinst, ut0); 
    else if (ut0 == FP_UT || ut1 == FP_UT) 
      setExprType(sinst, FP_UT); 
    else setExprType(sinst, NFP_UT); 
  }
  else {
    setExprType(sinst, llvmT2utccT(sinst->getType(), sinst)); 
  }
}


// -- handle phi node -- 
void UnsafeTypeCastingCheck::handlePHINode (Instruction *inst) {
  PHINode *pnode = dyn_cast<PHINode>(inst); 
  if (pnode == NULL) 
    utccAbort("handlePHINode cannot process with a non-phinode instruction");         
  assert(pnode->getNumIncomingValues() >= 1); 

  UTCC_TYPE pnode_ut = queryExprType(pnode->getIncomingValue(0)); 

  for (unsigned int i = 1 ; 
       i < pnode->getNumIncomingValues() ; 
       i++) {
    if (pnode_ut == UH_UT) break; 
    UTCC_TYPE next_ut = queryExprType(pnode->getIncomingValue(i)); 

    if (pnode_ut == UINT_UT) {
      if (next_ut == UINT_UT) ; 
      else if (next_ut == NINT_UT) pnode_ut = NINT_UT; 
      else if (next_ut == INT_UT) pnode_ut = INT_UT; 
      else pnode_ut = UH_UT; 
    }
    else if (pnode_ut == NINT_UT) {
      if (next_ut == UINT_UT || next_ut == NINT_UT) ; 
      else if (next_ut == INT_UT) pnode_ut = INT_UT; 
      else pnode_ut = UH_UT; 
    }
    else if (pnode_ut == INT_UT) {
      if (next_ut == UINT_UT || next_ut == NINT_UT || next_ut == INT_UT) ; 
      else pnode_ut = UH_UT; 
    }
    else if (pnode_ut == NFP_UT) {
      if (next_ut == NFP_UT) ; 
      else if (next_ut == FP_UT) pnode_ut = FP_UT; 
      else pnode_ut = UH_UT; 
    }
    else if (pnode_ut == FP_UT) {
      if (next_ut == NFP_UT || next_ut == FP_UT) ; 
      else pnode_ut = UH_UT; 
    }
    else utccAbort("Unknown Error on Setting next_ut in handlePHINode..."); 
  }

  setExprType(pnode, pnode_ut); 
}


// -- handle all instruction -- 
void UnsafeTypeCastingCheck::handleInstruction (Function &func, 
						Instruction *inst) {
  switch (inst->getOpcode()) {
  case Instruction::Alloca: 
    handleAllocInstruction(inst); 
    break; 
    
  case Instruction::Load: 
    handleLoadInstruction(inst); 
    break; 
    
  case Instruction::Store: 
    handleStoreInstruction(inst); 
    break; 
    
  case Instruction::Call: 
    handleCallInstruction(inst); 
    break; 
    
  case Instruction::FPToUI: 
    handleFP2UIInstruction(func, inst); 
    break; 
    
  case Instruction::FPToSI: 
    handleFP2SIInstruction(inst); 
    break; 
    
  case Instruction::UIToFP: 
    handleUI2FPInstruction(inst); 
    break; 
    
  case Instruction::SIToFP: 
    handleSI2FPInstruction(inst); 
    break; 
    
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem: 
    handleBinaryOperation(inst); 
    break; 
    
  case Instruction::FPExt:
    handleFPExtInstruction(inst); 
    break; 
    
  case Instruction::FPTrunc: 
    handleFPTruncInstruction(inst); 
    break; 

  case Instruction::SExt: 
    handleSExtInstruction(inst); 
    break; 

  case Instruction::Trunc: 
    handleTruncInstruction(inst); 
    break; 

  case Instruction::ZExt: 
    handleZExtInstruction(inst); 
    break; 

  case Instruction::GetElementPtr: 
    handleGetElementPtrInstruction(inst); 
    break; 

  case Instruction::ExtractValue: 
    handleExtractValueInstruction(inst); 
    break; 

  case Instruction::BitCast: 
    handleBitCastInstruction(inst); 
    break; 

  case Instruction::Select:
    handleSelectInstruction(inst); 
    break; 
    
  case Instruction::PHI: 
    handlePHINode(inst); 
    break; 
    
    // ignore cases 
  case Instruction::Ret: 
  case Instruction::ICmp: 
  case Instruction::FCmp: 
  case Instruction::Unreachable:
    break; 

  case Instruction::Br: 
    break; 
    
  default: 
    if (UTCC_WARNING_UNHANDLED_INST) 
      utccWarning("This instruction is not handled by handleInstruction", inst); 
    break; 
  }
}


bool UnsafeTypeCastingCheck::doInitialization(llvm::Module &mod) { 
  errs() << "initialization... \n"; 
  
  for (Module::global_iterator git = mod.global_begin() ;
       git != mod.global_end() ; 
       git++) {
    GlobalValue *gv = dyn_cast<GlobalValue>(git);
    string gv_name = gv->getName().str(); 
    if (gv_name.compare("blockDim") == 0 || 
	gv_name.compare("gridDim") == 0 || 
	gv_name.compare("blockIdx") == 0 || 
	gv_name.compare("threadIdx") == 0) {
      setPointedType(gv, UINT_UT); 
    }
  }

  // get utcc_assert function call (assertion) 
  code_modified = false; 
  assert_func = mod.getFunction("utcc_assert"); 
  assert(assert_func != NULL); 

  return false; 
}


bool UnsafeTypeCastingCheck::runOnFunction(Function &func) { 
  string func_name = func.getName(); 
  utccPrint("Traversing Function: "); 
  utccPrintl(func_name); 
  
  // -- some functions are skipped... -- 
  if (func_name.compare("claimNonNegativeInt") == 0 || 
      func_name.compare("claimNonNegativeUint") == 0 || 
      func_name.compare("claimNonNegativeFP32") == 0 || 
      func_name.compare("claimNonNegativeFP64") == 0 || 
      func_name.compare("ceil") == 0 || 
      func_name.compare("floor") == 0 || 
      func_name.compare("fabs") == 0 || 
      func_name.compare("sqrt") == 0 || 
      func_name.compare("exp") == 0 || 
      func_name.compare("min") == 0 || 
      func_name.compare("max") == 0 || 
      func_name.compare("utcc_assert") == 0) {
    utccPrintl("    skipped..."); 
    return false; 
  }

  // -- be silent on some functions -- 
  bool prev_UTCC_VERBOSE = UTCC_VERBOSE; 
  if (func_name.compare("computeBezierLinesCDP") == 0) {
    utccPrintl("    silent..."); 
    UTCC_VERBOSE = false; 
  }
  
  // -- handle arguments -- 
  for (Function::arg_iterator ait = func.arg_begin() ; 
       ait != func.arg_end() ; 
       ait++) {
    Value *arg = dyn_cast<Value>(ait); 
    assert(arg != NULL); 
    setExprType(arg, llvmT2utccT(arg->getType(), arg)); 
  }
  
  // -- traversing basic blocks -- 
  for (Function::iterator bbit = func.begin() ; 
       bbit != func.end() ; 
       bbit++) {
    // -- traversing instructions -- 
    for (BasicBlock::iterator iit = bbit->begin() ; 
	 iit != bbit->end() ; 
	 iit++) {
      if (UTCC_VERBOSE) {
	utccPrintl("-- get inst. --"); 
	utccPrintLLVMValue(iit); 
      }
      handleInstruction(func, iit); 
    }
  }

  UTCC_VERBOSE = prev_UTCC_VERBOSE; 
  
  // return false; 
  if (code_modified) {
    code_modified = false; 
    return true; 
  }
  else return false; 
}


char UnsafeTypeCastingCheck::ID = 0; 
static RegisterPass<UnsafeTypeCastingCheck> X("utcc", "Unsafe Type-casting Check", false, false); 
