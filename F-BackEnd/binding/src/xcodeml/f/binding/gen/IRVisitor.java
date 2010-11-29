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
package xcodeml.f.binding.gen;

import xcodeml.binding.*;

/**
 * @version XcodeML_F.rng 1.0 (Mon Nov 29 15:25:59 JST 2010)
 * @author  Relaxer 1.0 (http://www.relaxer.org)
 */
public interface IRVisitor {
    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfXcodeProgram visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfXcodeProgram visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfTypeTable visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfTypeTable visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFfunctionType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFfunctionType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfParams visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfParams visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfName visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfName visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFbasicType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFbasicType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfKind visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfKind visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFunctionCall visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFunctionCall visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfArguments visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfArguments visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFcharacterRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFcharacterRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfVarRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfVarRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFarrayRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFarrayRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfIndexRange visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfIndexRange visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLowerBound visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLowerBound visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfUpperBound visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfUpperBound visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfStep visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfStep visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfArrayIndex visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfArrayIndex visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFarrayConstructor visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFarrayConstructor visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFmemberRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFmemberRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFcoArrayRef visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFcoArrayRef visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfVar visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfVar visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFdoLoop visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFdoLoop visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfValue visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfValue visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfRepeatCount visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfRepeatCount visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfNamedValue visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfNamedValue visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFintConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFintConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFrealConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFrealConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFcharacterConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFcharacterConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFlogicalConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFlogicalConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFcomplexConstant visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFcomplexConstant visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFstructConstructor visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFstructConstructor visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfPlusExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfPlusExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfMulExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfMulExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogNEQExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogNEQExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogGEExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogGEExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogOrExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogOrExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfUserBinaryExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfUserBinaryExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfUserUnaryExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfUserUnaryExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFfunction visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFfunction visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfDivExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfDivExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogEQVExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogEQVExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfMinusExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfMinusExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFpowerExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFpowerExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFconcatExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFconcatExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogEQExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogEQExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogGTExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogGTExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogLEExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogLEExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogLTExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogLTExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogAndExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogAndExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogNEQVExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogNEQVExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfUnaryMinusExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfUnaryMinusExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLogNotExpr visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLogNotExpr visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfLen visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfLen visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfCoShape visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfCoShape visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFstructType visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFstructType visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfSymbols visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfSymbols visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfId visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfId visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfGlobalSymbols visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfGlobalSymbols visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfGlobalDeclarations visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfGlobalDeclarations visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFfunctionDefinition visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFfunctionDefinition visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfDeclarations visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfDeclarations visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFinterfaceDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFinterfaceDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFfunctionDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFfunctionDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFmoduleProcedureDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFmoduleProcedureDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfVarDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfVarDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFuseDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFuseDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfRename visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfRename visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFuseOnlyDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFuseOnlyDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfRenamable visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfRenamable visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfExternDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfExternDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFnamelistDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFnamelistDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfVarList visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfVarList visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFcommonDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFcommonDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFstructDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFstructDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFentryDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFentryDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFequivalenceDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFequivalenceDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFdataDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFdataDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfValueList visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfValueList visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFpragmaStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFpragmaStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfBody visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfBody visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFdoStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFdoStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFselectCaseStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFselectCaseStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFcaseLabel visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFcaseLabel visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfGotoStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfGotoStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFstopStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFstopStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFpauseStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFpauseStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfExprStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfExprStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFifStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFifStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfCondition visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfCondition visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfThen visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfThen visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfElse visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfElse visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFdoWhileStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFdoWhileStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFcycleStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFcycleStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFexitStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFexitStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfStatementLabel visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfStatementLabel visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFreadStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFreadStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfNamedValueList visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfNamedValueList visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFwriteStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFwriteStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFprintStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFprintStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFrewindStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFrewindStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFendFileStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFendFileStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFbackspaceStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFbackspaceStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFopenStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFopenStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFcloseStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFcloseStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFinquireStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFinquireStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFformatDecl visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFformatDecl visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFallocateStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFallocateStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfAlloc visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfAlloc visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFdeallocateStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFdeallocateStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFcontainsStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFcontainsStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfContinueStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfContinueStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFreturnStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFreturnStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFwhereStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFwhereStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFnullifyStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFnullifyStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfText visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfText visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFassignStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFassignStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFpointerAssignStatement visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFpointerAssignStatement visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFmoduleDefinition visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFmoduleDefinition visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFblockDataDefinition visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFblockDataDefinition visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfDefModelArraySubscriptSequence1 visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfDefModelArraySubscriptSequence1 visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFequivalenceDeclSequence visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFequivalenceDeclSequence visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFdataDeclSequence visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFdataDeclSequence visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfFdoStatementSequence visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfFdoStatementSequence visitable);

    /**
     * Visits this node for enter behavior.
     *
     * @param visitable
     * @return boolean
     */
    boolean enter(XbfGotoStatementSequence visitable);

    /**
     * Visits this node for leave behavior.
     *
     * @param visitable
     */
    void leave(XbfGotoStatementSequence visitable);
}
