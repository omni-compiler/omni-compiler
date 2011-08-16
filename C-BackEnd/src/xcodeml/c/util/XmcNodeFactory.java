/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.util;

import xcodeml.c.binding.gen.*;
import xcodeml.c.obj.*;

/**
 * Xmc Object factory.
 */
public class XmcNodeFactory extends AbstractXcodeML_CFactory
{
    @Override
    public XbcGccAlignOfExpr createXbcGccAlignOfExpr()
    {
        return new XmcGccAlignOfExpr();
    }

    @Override
    public XbcArguments createXbcArguments()
    {
        return new XmcArguments();
    }

    @Override
    public XbcArrayRef createXbcArrayRef()
    {
        return new XmcArrayRef();
    }

    @Override
    public XbcArrayType createXbcArrayType()
    {
        return new XmcArrayType();
    }

    @Override
    public XbcAsgBitAndExpr createXbcAsgBitAndExpr()
    {
        return new XmcAsgBitAndExpr();
    }

    @Override
    public XbcAsgBitOrExpr createXbcAsgBitOrExpr()
    {
        return new XmcAsgBitOrExpr();
    }

    @Override
    public XbcAsgBitXorExpr createXbcAsgBitXorExpr()
    {
        return new XmcAsgBitXorExpr();
    }

    @Override
    public XbcAsgDivExpr createXbcAsgDivExpr()
    {
        return new XmcAsgDivExpr();
    }

    @Override
    public XbcAsgLshiftExpr createXbcAsgLshiftExpr()
    {
        return new XmcAsgLshiftExpr();
    }

    @Override
    public XbcAsgMinusExpr createXbcAsgMinusExpr()
    {
        return new XmcAsgMinusExpr();
    }

    @Override
    public XbcAsgModExpr createXbcAsgModExpr()
    {
        return new XmcAsgModExpr();
    }

    @Override
    public XbcAsgMulExpr createXbcAsgMulExpr()
    {
        return new XmcAsgMulExpr();
    }

    @Override
    public XbcAsgPlusExpr createXbcAsgPlusExpr()
    {
        return new XmcAsgPlusExpr();
    }

    @Override
    public XbcAsgRshiftExpr createXbcAsgRshiftExpr()
    {
        return new XmcAsgRshiftExpr();
    }

    @Override
    public XbcAssignExpr createXbcAssignExpr()
    {
        return new XmcAssignExpr();
    }

    @Override
    public XbcBasicType createXbcBasicType()
    {
        return new XmcBasicType();
    }

    @Override
    public XbcBitAndExpr createXbcBitAndExpr()
    {
        return new XmcBitAndExpr();
    }

    @Override
    public XbcBitNotExpr createXbcBitNotExpr()
    {
        return new XmcBitNotExpr();
    }

    @Override
    public XbcBitOrExpr createXbcBitOrExpr()
    {
        return new XmcBitOrExpr();
    }

    @Override
    public XbcBitXorExpr createXbcBitXorExpr()
    {
        return new XmcBitXorExpr();
    }

    @Override
    public XbcBody createXbcBody()
    {
        return new XmcBody();
    }

    @Override
    public XbcBreakStatement createXbcBreakStatement()
    {
        return new XmcBreakStatement();
    }

    @Override
    public XbcCaseLabel createXbcCaseLabel()
    {
        return new XmcCaseLabel();
    }

    @Override
    public XbcCastExpr createXbcCastExpr()
    {
        return new XmcCastExpr();
    }

    @Override
    public XbcCommaExpr createXbcCommaExpr()
    {
        return new XmcCommaExpr();
    }

    @Override
    public XbcCompoundStatement createXbcCompoundStatement()
    {
        return new XmcCompoundStatement();
    }

    @Override
    public XbcCondExpr createXbcCondExpr()
    {
        return new XmcCondExpr();
    }

    @Override
    public XbcCondition createXbcCondition()
    {
        return new XmcCondition();
    }

    @Override
    public XbcContinueStatement createXbcContinueStatement()
    {
        return new XmcContinueStatement();
    }

    @Override
    public XbcDeclarations createXbcDeclarations()
    {
        return new XmcDeclarations();
    }

    @Override
    public XbcDefaultLabel createXbcDefaultLabel()
    {
        return new XmcDefaultLabel();
    }

    @Override
    public XbcDivExpr createXbcDivExpr()
    {
        return new XmcDivExpr();
    }

    @Override
    public XbcDoStatement createXbcDoStatement()
    {
        return new XmcDoStatement();
    }

    @Override
    public XbcElse createXbcElse()
    {
        return new XmcElse();
    }

    @Override
    public XbcEnumType createXbcEnumType()
    {
        return new XmcEnumType();
    }

    @Override
    public XbcExprStatement createXbcExprStatement()
    {
        return new XmcExprStatement();
    }

    @Override
    public XbcFloatConstant createXbcFloatConstant()
    {
        return new XmcFloatConstant();
    }

    @Override
    public XbcForStatement createXbcForStatement()
    {
        return new XmcForStatement();
    }

    @Override
    public XbcFuncAddr createXbcFuncAddr()
    {
        return new XmcFuncAddr();
    }

    @Override
    public XbcFunction createXbcFunction()
    {
        return new XmcFunction();
    }

    @Override
    public XbcFunctionCall createXbcFunctionCall()
    {
        return new XmcFunctionCall();
    }

    @Override
    public XbcFunctionDefinition createXbcFunctionDefinition()
    {
        return new XmcFunctionDefinition();
    }

    @Override
    public XbcFunctionType createXbcFunctionType()
    {
        return new XmcFunctionType();
    }

    @Override
    public XbcGccAsmDefinition createXbcGccAsmDefinition()
    {
        return new XmcGccAsmDefinition();
    }

    @Override
    public XbcGccAsmStatement createXbcGccAsmStatement()
    {
        return new XmcGccAsmStatement();
    }

    @Override
    public XbcGccAttributes createXbcGccAttributes()
    {
        return new XmcGccAttributes();
    }

    @Override
    public XbcGlobalDeclarations createXbcGlobalDeclarations()
    {
        return new XmcGlobalDeclarations();
    }

    @Override
    public XbcGlobalSymbols createXbcGlobalSymbols()
    {
        return new XmcGlobalSymbols();
    }

    @Override
    public XbcGotoStatement createXbcGotoStatement()
    {
        return new XmcGotoStatement();
    }

    @Override
    public XbcId createXbcId()
    {
        return new XmcId();
    }

    @Override
    public XbcIfStatement createXbcIfStatement()
    {
        return new XmcIfStatement();
    }

    @Override
    public XbcInit createXbcInit()
    {
        return new XmcInit();
    }

    @Override
    public XbcIntConstant createXbcIntConstant()
    {
        return new XmcIntConstant();
    }

    @Override
    public XbcIter createXbcIter()
    {
        return new XmcIter();
    }

    @Override
    public XbcLogAndExpr createXbcLogAndExpr()
    {
        return new XmcLogAndExpr();
    }

    @Override
    public XbcLogEQExpr createXbcLogEQExpr()
    {
        return new XmcLogEQExpr();
    }

    @Override
    public XbcLogGEExpr createXbcLogGEExpr()
    {
        return new XmcLogGEExpr();
    }

    @Override
    public XbcLogGTExpr createXbcLogGTExpr()
    {
        return new XmcLogGTExpr();
    }

    @Override
    public XbcLogLEExpr createXbcLogLEExpr()
    {
        return new XmcLogLEExpr();
    }

    @Override
    public XbcLogLTExpr createXbcLogLTExpr()
    {
        return new XmcLogLTExpr();
    }

    @Override
    public XbcLogNEQExpr createXbcLogNEQExpr()
    {
        return new XmcLogNEQExpr();
    }

    @Override
    public XbcLogNotExpr createXbcLogNotExpr()
    {
        return new XmcLogNotExpr();
    }

    @Override
    public XbcLogOrExpr createXbcLogOrExpr()
    {
        return new XmcLogOrExpr();
    }

    @Override
    public XbcLonglongConstant createXbcLonglongConstant()
    {
        return new XmcLonglongConstant();
    }

    @Override
    public XbcLshiftExpr createXbcLshiftExpr()
    {
        return new XmcLshiftExpr();
    }

    @Override
    public XbcMemberAddr createXbcMemberAddr()
    {
        return new XmcMemberAddr();
    }

    @Override
    public XbcMemberArrayRef createXbcMemberArrayRef()
    {
        return new XmcMemberArrayRef();
    }

    @Override
    public XbcMemberRef createXbcMemberRef()
    {
        return new XmcMemberRef();
    }

    @Override
    public XbcMinusExpr createXbcMinusExpr()
    {
        return new XmcMinusExpr();
    }

    @Override
    public XbcModExpr createXbcModExpr()
    {
        return new XmcModExpr();
    }

    @Override
    public XbcMoeConstant createXbcMoeConstant()
    {
        return new XmcMoeConstant();
    }

    @Override
    public XbcMulExpr createXbcMulExpr()
    {
        return new XmcMulExpr();
    }

    @Override
    public XbcName createXbcName()
    {
        return new XmcName();
    }

    @Override
    public XbcParams createXbcParams()
    {
        return new XmcParams();
    }

    @Override
    public XbcPlusExpr createXbcPlusExpr()
    {
        return new XmcPlusExpr();
    }

    @Override
    public XbcPointerRef createXbcPointerRef()
    {
        return new XmcPointerRef();
    }

    @Override
    public XbcPointerType createXbcPointerType()
    {
        return new XmcPointerType();
    }

    @Override
    public XbcPostDecrExpr createXbcPostDecrExpr()
    {
        return new XmcPostDecrExpr();
    }

    @Override
    public XbcPostIncrExpr createXbcPostIncrExpr()
    {
        return new XmcPostIncrExpr();
    }

    @Override
    public XbcPreDecrExpr createXbcPreDecrExpr()
    {
        return new XmcPreDecrExpr();
    }

    @Override
    public XbcPreIncrExpr createXbcPreIncrExpr()
    {
        return new XmcPreIncrExpr();
    }

    @Override
    public XbcReturnStatement createXbcReturnStatement()
    {
        return new XmcReturnStatement();
    }

    @Override
    public XbcRshiftExpr createXbcRshiftExpr()
    {
        return new XmcRshiftExpr();
    }

    @Override
    public XbcStatementLabel createXbcStatementLabel()
    {
        return new XmcStatementLabel();
    }

    @Override
    public XbcStringConstant createXbcStringConstant()
    {
        return new XmcStringConstant();
    }

    @Override
    public XbcStructType createXbcStructType()
    {
        return new XmcStructType();
    }

    @Override
    public XbcSwitchStatement createXbcSwitchStatement()
    {
        return new XmcSwitchStatement();
    }

    @Override
    public XbcSymbols createXbcSymbols()
    {
        return new XmcSymbols();
    }

    @Override
    public XbcThen createXbcThen()
    {
        return new XmcThen();
    }

    @Override
    public XbcTypeTable createXbcTypeTable()
    {
        return new XmcTypeTable();
    }

    @Override
    public XbcUnaryMinusExpr createXbcUnaryMinusExpr()
    {
        return new XmcUnaryMinusExpr();
    }

    @Override
    public XbcUnionType createXbcUnionType()
    {
        return new XmcUnionType();
    }

    @Override
    public XbcValue createXbcValue()
    {
        return new XmcValue();
    }

    @Override
    public XbcVar createXbcVar()
    {
        return new XmcVar();
    }

    @Override
    public XbcVarAddr createXbcVarAddr()
    {
        return new XmcVarAddr();
    }

    @Override
    public XbcVarDecl createXbcVarDecl()
    {
        return new XmcVarDecl();
    }

    @Override
    public XbcWhileStatement createXbcWhileStatement()
    {
        return new XmcWhileStatement();
    }

    @Override
    public XbcXcodeProgram createXbcXcodeProgram()
    {
        return new XmcXcodeProgram();
    }

    @Override
    public XbcArrayAddr createXbcArrayAddr()
    {
        return new XmcArrayAddr();
    }

    @Override
    public XbcSizeOfExpr createXbcSizeOfExpr()
    {
        return new XmcSizeOfExpr();
    }


    @Override
    public XbcBuiltinOp createXbcBuiltinOp()
    {
        return new XmcBuiltinOp();
    }

    @Override
    public XbcGccLabelAddr createXbcGccLabelAddr()
    {
        return new XmcGccLabelAddr();
    }

    @Override
    public XbcMemberArrayAddr createXbcMemberArrayAddr()
    {
        return new XmcMemberArrayAddr();
    }

    @Override
    public XbcPragma createXbcPragma()
    {
        return new XmcPragma();
    }

    @Override
    public XbcText createXbcText()
    {
        return new XmcText();
    }
    
    @Override
    public XbcFunctionDecl createXbcFunctionDecl()
    {
        return new XbcFunctionDecl();
    }
    
    @Override
    public XbcGccMemberDesignator createXbcGccMemberDesignator()
    {
        return new XmcGccMemberDesignator();
    }

    @Override
    public XbcTypeName createXbcTypeName()
    {
        return new XmcTypeName();
    }

    @Override
    public XbcArraySize createXbcArraySize()
    {
        return new XmcArraySize();
    }


    @Override
    public XbcBitField createXbcBitField()
    {
        return new XmcBitField();
    }
    
    @Override
    public XbcGccAsmClobbers createXbcGccAsmClobbers()
    {
        return new XmcGccAsmClobbers();
    }

    @Override
    public XbcGccAsmOperand createXbcGccAsmOperand()
    {
        return new XmcGccAsmOperand();
    }

    @Override
    public XbcGccAsmOperands createXbcGccAsmOperands()
    {
        return new XmcGccAsmOperands();
    }

    @Override
    public XbcCompoundValue createXbcCompoundValue()
    {
        return new XmcCompoundValue();
    }

    @Override
    public XbcGccAsm createXbcGccAsm()
    {
        return new XmcGccAsm();
    }

    @Override
    public XbcGccCompoundExpr createXbcGccCompoundExpr()
    {
        return new XmcGccCompoundExpr();
    }

    @Override
    public XbcGccRangedCaseLabel createXbcGccRangedCaseLabel()
    {
        return new XmcGccRangedCaseLabel();
    }


    @Override
    public XbcGccAttribute createXbcGccAttribute()
    {
        return new XmcGccAttribute();
    }

    @Override
    public XbcCompoundValueExpr createXbcCompoundValueExpr()
    {
        return new XmcCompoundValueExpr();
    }

    @Override
    public XbcCompoundValueAddrExpr createXbcCompoundValueAddrExpr()
    {
        return new XmcCompoundValueAddrExpr();
    }
   
    @Override
    public XbcDesignatedValue createXbcDesignatedValue()
    {
        return new XmcDesignatedValue();
    }
    
    @Override
    public XbcCoArrayRef createXbcCoArrayRef()
    {
        return new XmcCoArrayRef();
    }

    @Override
    public XbcCoArrayType createXbcCoArrayType()
    {
        return new XmcCoArrayType();
    }

    @Override
    public XbcSubArrayRef createXbcSubArrayRef()
    {
        return new XmcSubArrayRef();
    }

    @Override
    public XbcIndexRange createXbcIndexRange()
    {
        return new XmcIndexRange();
    }

    @Override
    public XbcLowerBound createXbcLowerBound()
    {
        return new XmcLowerBound();
    }

    @Override
    public XbcUpperBound createXbcUpperBound()
    {
        return new XmcUpperBound();
    }

    @Override
    public XbcStep createXbcStep()
    {
        return new XmcStep();
    }

//     @Override
//     public XbcSubArrayRefLowerBound createXbcSubArrayRefLowerBound()
//     {
//         return new XmcSubArrayRefLowerBound();
//     }

//     @Override
//     public XbcSubArrayRefStep createXbcSubArrayRefStep()
//     {
//         return new XmcSubArrayRefStep();
//     }

//     @Override
//     public XbcSubArrayRefUpperBound createXbcSubArrayRefUpperBound()
//     {
//         return new XmcSubArrayRefUpperBound();
//     }

    @Override
    public XbcCoArrayAssignExpr createXbcCoArrayAssignExpr()
    {
        return new XmcCoArrayAssignExpr();
    }
}
