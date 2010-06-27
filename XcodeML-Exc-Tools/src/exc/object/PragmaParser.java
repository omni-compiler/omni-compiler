/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import exc.openmp.OMPpragmaParser;
import xcodeml.XmException;

import java.util.Stack;

/**
 * Base pragma parser.
 */
public class PragmaParser
{
    /** XobjectFile of target AST */
    private XobjectFile xobjFile;
    
    private OMPpragmaParser ompPragmaParser;
    
    /** A stack of environments. */
    protected Stack<XobjList> stackEnv = new Stack<XobjList>();

    public PragmaParser(XobjectFile xobjFile)
    {
        this.xobjFile = xobjFile;
        ompPragmaParser = new OMPpragmaParser(this);
        pushEnv((XobjList)xobjFile.getGlobalIdentList());
    }

    public void pushEnv(XobjList v)
    {
        stackEnv.push(v);
    }

    public void popEnv()
    {
        stackEnv.pop();
    }

    public XobjList peekEnv()
    {
        return stackEnv.peek();
    }
    
    public Ident findIdent(String name, int find_kind)
    {
        for(int i = stackEnv.size() - 1; i >=0; --i) {
            XobjList id_list = stackEnv.get(i);
            Ident id = id_list.findIdent(name, find_kind);
            if(id != null)
                return id;
        }
        
        return null;
    }

    public class Result
    {
        public final PragmaSyntax pragma_syntax;
        public final Xobject xobject;

        public Result(PragmaSyntax pragma_syntax, Xobject obj)
        {
            this.pragma_syntax = pragma_syntax;
            this.xobject = obj;
        }
    };
    
    public XobjectFile getXobjectFile()
    {
        return xobjFile;
    }
    
    public Xobject parse(Xobject x) throws XmException
    {
        switch(x.Opcode()) {
        case OMP_PRAGMA:
            return ompPragmaParser.parse(x);
        case XMP_PRAGMA:
            // do nothing for XcalableMP directives
            return x;
        }
        return x;
    }
    
    public boolean isPrePostPair(Xobject prefix, Xobject postfix)
    {
        switch(prefix.Opcode()) {
        case OMP_PRAGMA:
            return ompPragmaParser.isPrePostPair(prefix, postfix);
        }
        return false;
    }
    
    public void completePragmaEnd(Xobject prefix, Xobject body)
    {
        switch(prefix.Opcode()) {
        case OMP_PRAGMA:
            ompPragmaParser.completePragmaEnd(prefix, body);
            break;
        }
    }
    
    public XobjArgs getAbbrevPostfix(XobjArgs prefixArgs)
    {
        switch(prefixArgs.getArg().Opcode()) {
        case OMP_PRAGMA:
            return ompPragmaParser.getAbbrevPostfix(prefixArgs);
        }
        return null;
    }
    
    public void mergeStartAndPostfixArgs(Xobject start, Xobject postfix)
    {
        switch(start.Opcode()) {
        case OMP_PRAGMA:
            ompPragmaParser.mergeStartAndPostfixArgs(start, postfix);
        }
    }
}
