/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.c.decompile;

import xcodeml.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents following elements:
 *   ifStatement, forStatement, whileStatement, doStatement, switchStatement,
 *   returnStatement, continueStatement, breakStatement, gotoStatement,
 *   caseLabel, defaultLabel, statementLabel
 */
public abstract class XcControlStmtObj extends XcStmtObj
{
    private XcControlStmtEnum _controlStmtEnum;

    private XcControlStmtObj(XcControlStmtEnum controlStmtEnum)
    {
        _controlStmtEnum = controlStmtEnum;
    }

    /**
     * Checks what the object is.
     * 
     * @return a member of enumerator of the control statement object. 
     */
    public final XcControlStmtEnum getControlStmtEnum()
    {
        return _controlStmtEnum;
    }

    protected boolean _isAssignExpr(XcExprObj expr)
    {
        if((expr instanceof XcOperatorObj) == false)
            return false;

        return ((XcOperatorObj)expr).isAssignExpr();
    }

    private static abstract class Conditional extends XcControlStmtObj
    {
        private XcExprObj _condExpr;
        
        public Conditional(XcControlStmtEnum controlStmtEnum)
        {
            super(controlStmtEnum);
        }
        
        public final void setCondExpr(XcExprObj condExpr)
        {
            _condExpr = condExpr;
        }
        
        public final XcExprObj getCondExpr()
        {
            return _condExpr;
        }

        @Override
        public void checkChild()
        {
            if(_condExpr == null)
                throw new IllegalArgumentException("no condition");
        }
    }

    /**
     * Internal object represents ifStatement.
     */
    public static final class If extends Conditional
    {
        private XcStmtObj _thenStmt, _elseStmt;
     
        /**
         * Creates If.
         */
        public If()
        {
            super(XcControlStmtEnum.IF);
        }

        /**
         * Sets the statement object to then part.
         * 
         * @param thenStmt the statement object of then part.
         */
        public final void setThenStmt(XcStmtObj thenStmt)
        {
            _thenStmt = thenStmt;
        }
        
        /**
         * Sets the statement object to else part.
         * 
         * @param thenStmt the statement object of else part.
         */
        public final void setElseStmt(XcStmtObj elseStmt)
        {
            _elseStmt = elseStmt;
        }
        
        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj)
                setCondExpr((XcExprObj)child);
            else if(child instanceof XcStmtObj) {
                if(_thenStmt == null) {
                    setThenStmt((XcStmtObj)child);
                } else if(_elseStmt == null) {
                    setElseStmt((XcStmtObj)child);
                } else
                    throw new IllegalArgumentException(child.getClass().getName());
            } else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(getCondExpr(), _thenStmt, _elseStmt);
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                setCondExpr((XcExprObj)child);
                break;
            case 1:
                _thenStmt = (XcStmtObj)child;
                break;
            case 2:
                _elseStmt = (XcStmtObj)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            boolean braced = _isAssignExpr(getCondExpr());

            w.add("if");
            if(braced)
                w.add("(");
            w.addBrace(getCondExpr());
            if(braced)
                w.add(")");

            if(_thenStmt != null)
                w.addSpc(_thenStmt);
            else
                w.eos();

            if(_elseStmt != null)
                w.add("else ").add(_elseStmt);
        }
    }

    /**
     * Internal object represents whileStatement.
     */
    public static final class While extends Conditional
    {
        private XcStmtObj _stmt;

        /**
         * Creates While.
         */
        public While()
        {
            super(XcControlStmtEnum.WHILE);
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj)
                setCondExpr((XcExprObj)child);
            else if(child instanceof XcStmtObj)
                _stmt = (XcStmtObj)child;
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(getCondExpr(), _stmt);
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                setCondExpr((XcExprObj)child);
                break;
            case 1:
                _stmt = (XcStmtObj)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }
        
        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            boolean braced = _isAssignExpr(getCondExpr());
            
            w.add("while");
            if(braced)
                w.add("(");
            w.addBrace(getCondExpr());
            if(braced)
                w.add(")");

            if(_stmt != null)
                w.addSpc(_stmt);
            else
                w.eos();
        }
    }
    
    /**
     *  Internal ojbect represents doStatement.
     */
    public static final class Do extends Conditional
    {
        private XcStmtObj _stmt;

        /**
         * Creates Do.
         */
        public Do()
        {
            super(XcControlStmtEnum.DO);
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj)
                setCondExpr((XcExprObj)child);
            else if(child instanceof XcStmtObj)
                _stmt = (XcStmtObj)child;
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(getCondExpr(), _stmt);
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                setCondExpr((XcExprObj)child);
                break;
            case 1:
                _stmt = (XcStmtObj)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            boolean braced = _isAssignExpr(getCondExpr());

            w.add("do");
            if(_stmt != null)
                w.addSpc(_stmt);
            else
                w.add(";");

            w.addSpc("while");

            if(braced)
                w.add("(");
            w.addBrace(getCondExpr());
            if(braced)
                w.add(")");

            w.eos();
        }
    }
    
    /**
     *  Internal object represents forStatement.
     */
    public static final class For extends XcControlStmtObj
    {
        private XcExprObj _initDecls, _condExpr, _incrExpr;

        private XcStmtObj _stmt;

        /**
         * Creates For.
         */
        public For()
        {
            super(XcControlStmtEnum.FOR);
        }

        /**
         *  Sets an expression object to an init part.
         *  
         * @param initDecls an expression object of an init part.
         */
        public final void setInitDecls(XcExprObj initDecls)
        {
            _initDecls = initDecls;
        }

        /**
         *  Sets an expression object to a condition part.
         *  
         * @param condExpr an expression object of a condition part.
         */
        public final void setCondExpr(XcExprObj condExpr)
        {
            _condExpr = condExpr;
        }
        
        /**
         *  Sets an expression object to an increment part.
         * 
         * @param incrExpr an expression object of an increment part.
         */
        public final void setIncrExpr(XcExprObj incrExpr)
        {
            _incrExpr = incrExpr;
        }
        
        /**
         * Sets a statement object to a body of for statement.
         * 
         * @param stmt a body statement of a for statement.
         */
        public final void setStmt(XcStmtObj stmt)
        {
            _stmt = stmt;
        }
        
        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcStmtObj) {
                _stmt = (XcStmtObj)child;
            } else if(child instanceof XcNullExpr) {
                if (_initDecls == null) {
                    _initDecls = (XcNullExpr)child;
                } else if(_condExpr == null) {
                    _condExpr = (XcNullExpr)child;
                } else if(_incrExpr == null) {
                    _incrExpr = (XcNullExpr)child;
                } else
                    throw new IllegalArgumentException(child.getClass().getName());
            } else if(child instanceof XcExprObj) {
                if (_initDecls == null) {
                    _initDecls = (XcExprObj)child;
                } else if(_condExpr == null) {
                    _condExpr = (XcExprObj)child;
                } else if(_incrExpr == null) {
                    _incrExpr = (XcExprObj)child;
                } else
                    throw new IllegalArgumentException(child.getClass().getName());
            } else 
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(_initDecls, _condExpr, _incrExpr, _stmt);
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                _initDecls = (XcExprObj)child;
                break;
            case 1:
                _condExpr = (XcExprObj)child;
                break;
            case 2:
                _incrExpr = (XcExprObj)child;
                break;
            case 3:
                _stmt = (XcStmtObj)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }

        @Override
        public void checkChild()
        {
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);
            boolean brace = _isAssignExpr(_condExpr);

            w.add("for(").addSpc(_initDecls);

            w.add(";");
            if(brace)
                w.add("(");
            w.addSpc(_condExpr);
            if(brace)
                w.add(")");
            w.add(";");

            w.addSpc(_incrExpr).add(")");

            if(_stmt != null)
                w.addSpc(_stmt);
            else
                w.eos();
        }
    }
    
    /**
     * Internal object represent switchStatement.
     */
    public static final class Switch extends Conditional
    {
        private XcStmtObj _stmt;
        
        /**
         * Creates Switch.
         */
        public Switch()
        {
            super(XcControlStmtEnum.SWITCH);
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj)
                setCondExpr((XcExprObj)child);
            else if(child instanceof XcStmtObj)
                _stmt = (XcStmtObj)child;
            else
                throw new IllegalArgumentException();
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(getCondExpr(), _stmt);
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                setCondExpr((XcExprObj)child);
                break;
            case 1:
                _stmt = (XcStmtObj)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }
        
        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);
            
            w.add("switch").addBrace(getCondExpr()).add(_stmt);
        }
    }
    
    /**
     * Internal object represents returnStatement.
     */
    public static final class Return extends XcControlStmtObj
    {
        private XcExprObj _expr;

        /**
         * Creates Retrun.
         */
        public Return()
        {
            this(null);
        }

        /**
         *  Creates Return.
         * 
         * @param expr an expression returned by the statement.
         */
        public Return(XcExprObj expr)
        {
            super(XcControlStmtEnum.RETURN);
            _expr = expr;
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj)
                _expr = (XcExprObj)child;
            else
                throw new IllegalArgumentException();
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(_expr);
        }

        @Override
        public void checkChild()
        {
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                _expr = (XcExprObj)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }
        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);
            
            w.add("return");
            if(_expr != null)
                w.add(" ").add(_expr).eos();
            else
                w.eos();
        }
    }
    
    /**
     * Extends this object bans to have any child.
     */
    public static abstract class Leaf extends XcControlStmtObj
    {
        private Leaf(XcControlStmtEnum controlStmtEnum)
        {
            super(controlStmtEnum);
        }
        
        @Override
        public void addChild(XcNode child)
        {
            throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public XcNode[] getChild()
        {
            return null;
        }

        @Override
        public void checkChild()
        {
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }
    
    /**
     * Internal object represents continueStatement.
     */
    public static final class Continue extends Leaf
    {
        /**
         * Creates Continue.
         */
        public Continue()
        {
            super(XcControlStmtEnum.CONTINUE);
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            w.add("continue").eos();
        }
    }

    /**
     * Internal object represents breakStatement.
     */
    public static final class Break extends Leaf
    {
        /**
         * Creates Break.
         */
        public Break()
        {
            super(XcControlStmtEnum.BREAK);
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            w.add("break").eos();
        }
    }

    /**
     * Internal object represent defalutLabel.
     */
    public static final class DefaultLabel extends Leaf
    {
        /**
         * Creates DefaultLabel.
         */
        public DefaultLabel()
        {
            super(XcControlStmtEnum.DEFAULT_LABEL);
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            w.add("default:");
        }
    }

    /**
     *  Internal object represents caseLabel.
     */
    public static final class CaseLabel extends XcControlStmtObj
    {
        private XcExprObj _value;

        /**
         * Creates CaseLabel.
         */
        public CaseLabel()
        {
            this(null);
        }

        /**
         * Creates CaseLabel.
         * 
         * @param value a value of label. 
         */
        public CaseLabel(XcExprObj value)
        {
            super(XcControlStmtEnum.CASE_LABEL);
            _value = value;
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj)
                _value = (XcExprObj)child;
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(_value);
        }

        @Override
        public void checkChild()
        {
            if(_value == null)
                throw new IllegalArgumentException("no case label");
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                _value = (XcExprObj)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            w.add("case ");
            w.add(_value);
            w.add(":");
        }
    }

    /**
     * Internal object represents statementLabel.
     */
    public static final class Label extends XcControlStmtObj
    {
        private String _name;

        /**
         *  Creates Label
         *  
         * @param name a name of label.
         */
        public Label(String name)
        {
            super(XcControlStmtEnum.STMT_LABEL);
            _name = name;
        }

        @Override
        public void addChild(XcNode child)
        {
            throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public XcNode[] getChild()
        {
            return null;
        }

        @Override
        public void checkChild()
        {
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            w.add(_name).addSpc(":");
        }
    }

    /**
     * Internal object represents gotoStatement.
     */
    public static final class Goto extends XcControlStmtObj
    {
        private XcNameObj _addr;

        /**
         *  Creates Goto.
         */
        public Goto()
        {
            super(XcControlStmtEnum.GOTO);
        }

        /**
         * Gets a name of a label the goto object pointed.
         * 
         * @return a name of a label the goto object pointed.
         */
        public final XcNameObj getAddr()
        {
            return _addr;
        }

        /**
         * Sets a name of a label the goto object pointed.
         * 
         * @param addr a name of a label the goto object pointed.
         */
        public final void setAddr(XcNameObj addr)
        {
            _addr = addr;
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcNameObj)
                _addr = (XcNameObj)child;
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                _addr = (XcNameObj)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(_addr);
        }

        @Override
        public void checkChild()
        {
            if(_addr == null)
                throw new IllegalArgumentException("no label name or address expression");
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            w.add("goto ").add(_addr).eos();
        }
    }

    /**
     * Internal object represent gccRangedCaseLabel.
     */
    public static final class GccRangedCaseLabel extends XcControlStmtObj
    {
        private XcExprObj _lowerValue;
        private XcExprObj _upperValue;

        /**
         * Creates GccRangedCaseLabel.
         */
        public GccRangedCaseLabel()
        {
            super(XcControlStmtEnum.GCC_RANGED_CASE_LABEL);
        }

        /**
         * Creates GccRangedCaseLabel.
         * 
         * @param lowerValue a lower bound value label.
         * @param upperValue a upper bound value label.
         */
        public GccRangedCaseLabel(XcExprObj lowerValue,
                                  XcExprObj upperValue)
        {
            super(XcControlStmtEnum.GCC_RANGED_CASE_LABEL);
            _lowerValue = lowerValue;
            _upperValue = upperValue;
        }

        @Override
        public void addChild(XcNode child)
        {
            if(child instanceof XcExprObj) {
                if(_lowerValue == null) {
                    _lowerValue = (XcExprObj)child;
                } else {
                    _upperValue = (XcExprObj)child;
                }
            }
            else
                throw new IllegalArgumentException(child.getClass().getName());
        }

        @Override
        public XcNode[] getChild()
        {
            return toNodeArray(_lowerValue, _upperValue);
        }

        @Override
        public void checkChild()
        {
            if((_lowerValue == null) || (_upperValue == null))
                throw new IllegalArgumentException("no case label");
        }

        @Override
        public final void setChild(int index, XcNode child)
        {
            switch(index) {
            case 0:
                _lowerValue = (XcExprObj)child;
                break;
            case 1:
                _upperValue = (XcExprObj)child;
                break;
            default:
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
            }
        }

        @Override
        public final void appendCode(XmcWriter w) throws XmException
        {
            super.appendCode(w);

            w.add("case ");
            w.add(_lowerValue);
            w.add(" ... ");
            w.add(_upperValue);
            w.add(":");
        }
    }

    @Override
    public String toString()
    {
        return _controlStmtEnum.toString();
    }
}
