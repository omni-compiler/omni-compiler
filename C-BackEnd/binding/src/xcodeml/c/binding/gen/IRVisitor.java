/*
 * The Relaxer artifact
 * Copyright (c) 2000-2003, ASAMI Tomoharu, All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer. 
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package xcodeml.c.binding.gen;

import xcodeml.binding.*;

/**
 * @version XcodeML_C.rng 1.0 (Thu Sep 24 16:30:21 JST 2009)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public interface IRVisitor {
    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcXcodeProgram visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcXcodeProgram visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcTypeTable visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcTypeTable visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcArrayType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcArrayType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccAttributes visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccAttributes visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccAttribute visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccAttribute visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcSubArrayRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcSubArrayRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcSubArrayRefLowerBound visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcSubArrayRefLowerBound visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcSubArrayRefUpperBound visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcSubArrayRefUpperBound visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcSubArrayRefStep visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcSubArrayRefStep visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcBuiltinOp visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcBuiltinOp visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcFunctionCall visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcFunctionCall visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcFunction visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcFunction visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcArguments visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcArguments visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccCompoundExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccCompoundExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCompoundStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCompoundStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcSymbols visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcSymbols visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcId visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcId visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcName visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcName visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcValue visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcValue visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCastExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCastExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCoArrayRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCoArrayRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcStringConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcStringConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcVar visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcVar visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcArrayRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcArrayRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcArrayAddr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcArrayAddr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCompoundValueExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCompoundValueExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCompoundValueAddrExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCompoundValueAddrExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcVarAddr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcVarAddr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCommaExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCommaExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcIntConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcIntConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcFloatConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcFloatConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLonglongConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLonglongConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcMoeConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcMoeConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcFuncAddr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcFuncAddr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcSizeOfExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcSizeOfExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCondExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCondExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcBitXorExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcBitXorExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLogGEExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLogGEExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAssignExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAssignExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcMemberRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcMemberRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcPointerRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcPointerRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcBitNotExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcBitNotExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcPreIncrExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcPreIncrExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccAlignOfExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccAlignOfExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccLabelAddr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccLabelAddr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCoArrayAssignExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCoArrayAssignExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcDivExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcDivExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcRshiftExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcRshiftExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgDivExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgDivExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLogAndExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLogAndExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcUnaryMinusExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcUnaryMinusExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcMemberAddr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcMemberAddr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcMemberArrayRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcMemberArrayRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcMemberArrayAddr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcMemberArrayAddr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcTypeName visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcTypeName visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcPlusExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcPlusExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcMinusExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcMinusExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcMulExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcMulExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcModExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcModExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLshiftExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLshiftExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcBitAndExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcBitAndExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcBitOrExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcBitOrExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgPlusExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgPlusExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgMinusExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgMinusExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgMulExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgMulExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgModExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgModExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgLshiftExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgLshiftExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgRshiftExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgRshiftExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgBitAndExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgBitAndExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgBitOrExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgBitOrExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcAsgBitXorExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcAsgBitXorExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLogEQExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLogEQExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLogNEQExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLogNEQExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLogGTExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLogGTExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLogLEExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLogLEExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLogLTExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLogLTExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLogOrExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLogOrExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcLogNotExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcLogNotExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcPostIncrExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcPostIncrExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcPostDecrExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcPostDecrExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcPreDecrExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcPreDecrExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcDesignatedValue visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcDesignatedValue visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCompoundValue visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCompoundValue visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcBitField visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcBitField visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcPragma visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcPragma visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcText visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcText visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcDeclarations visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcDeclarations visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcFunctionDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcFunctionDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccAsm visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccAsm visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcVarDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcVarDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcFunctionDefinition visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcFunctionDefinition visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcParams visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcParams visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcBody visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcBody visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcForStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcForStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcInit visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcInit visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCondition visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCondition visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcIter visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcIter visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcIfStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcIfStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcThen visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcThen visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcElse visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcElse visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcWhileStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcWhileStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcDoStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcDoStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcSwitchStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcSwitchStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccRangedCaseLabel visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccRangedCaseLabel visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCaseLabel visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCaseLabel visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccAsmStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccAsmStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccAsmOperands visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccAsmOperands visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccAsmOperand visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccAsmOperand visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccAsmClobbers visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccAsmClobbers visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcStatementLabel visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcStatementLabel visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcBreakStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcBreakStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcContinueStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcContinueStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcReturnStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcReturnStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGotoStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGotoStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcDefaultLabel visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcDefaultLabel visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcExprStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcExprStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccMemberDesignator visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccMemberDesignator visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcArraySize visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcArraySize visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcFunctionType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcFunctionType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcBasicType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcBasicType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcPointerType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcPointerType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcUnionType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcUnionType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcStructType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcStructType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcEnumType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcEnumType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcCoArrayType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcCoArrayType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGlobalSymbols visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGlobalSymbols visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGlobalDeclarations visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGlobalDeclarations visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbcGccAsmDefinition visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbcGccAsmDefinition visitable);
}
