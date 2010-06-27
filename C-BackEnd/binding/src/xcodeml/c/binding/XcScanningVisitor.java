/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.binding;

import xcodeml.XmObj;
import xcodeml.binding.IRNode;
import xcodeml.c.binding.gen.*;

/**
 * A Visitor oly enter children recursivele, and do nothing other.
 */
public class XcScanningVisitor extends RVisitorBase
{
    /**
     * return OR value of all children, if child is leaf then return true.
     *
     * @rertun OR value of all children.
     */
    protected boolean _enterChildren(IRNode visitable)
    {
        if(visitable == null)
            return false;

        for(IRNode child : visitable.rGetRNodes()) {
            if(((IRVisitable)child).enter(this) == true)
                return true;
        }

        return false;
    }

    @Override
    public boolean enter(XbcXcodeProgram visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcTypeTable visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcFunctionType visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccAttributes visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcBuiltinOp visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCondExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcFunctionCall visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcFunction visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcArguments visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccCompoundExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCompoundStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcSymbols visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcId visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcName visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcValue visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcMemberRef visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcMemberArrayRef visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCastExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcMemberArrayAddr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcMemberAddr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcVarAddr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcArrayAddr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcStringConstant visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcVar visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcArrayRef visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCommaExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcSizeOfExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcBitNotExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcIntConstant visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcFloatConstant visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcFuncAddr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcUnaryMinusExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLogNotExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCompoundValueExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcMinusExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcBitOrExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcBitXorExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgPlusExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgMulExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgDivExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgRshiftExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgBitOrExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgBitXorExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLogEQExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLogLEExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLogAndExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLogOrExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcPlusExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLshiftExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgMinusExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgModExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgBitAndExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLogLTExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcDesignatedValue visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcPreDecrExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccAlignOfExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCompoundValue visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLonglongConstant visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcMoeConstant visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcPointerRef visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcMulExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcDivExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcRshiftExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcBitAndExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAsgLshiftExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLogNEQExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLogGEExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcLogGTExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcPostDecrExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcPreIncrExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccLabelAddr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCompoundValueAddrExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcAssignExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcModExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcPostIncrExpr visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcTypeName visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcBitField visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcText visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcPragma visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcDeclarations visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcFunctionDefinition visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcParams visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcBody visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcForStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcInit visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCondition visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcIter visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcIfStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcThen visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcElse visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcDoStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccRangedCaseLabel visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccAsmStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccAsmOperands visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccAsmOperand visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccAsmClobbers visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcReturnStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcStatementLabel visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcContinueStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGotoStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcWhileStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcSwitchStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcExprStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCaseLabel visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcBreakStatement visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcDefaultLabel visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcVarDecl visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccAsm visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcFunctionDecl visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccMemberDesignator visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccAttribute visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcArrayType visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcArraySize visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcBasicType visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcPointerType visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcStructType visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcUnionType visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcEnumType visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGlobalSymbols visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGlobalDeclarations visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcGccAsmDefinition visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCoArrayRef visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcCoArrayType visitable)
    {
        return _enterChildren((IRNode)visitable);
    }

    @Override
    public boolean enter(XbcSubArrayRef visitable)
    {
        return _enterChildren((IRNode)visitable);
    }
}