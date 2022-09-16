/* -*- Mode: java; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
package exc.object;

// import exc.openmp.OMPpragmaParser;
import xcodeml.util.XmException;
import java.util.Stack;

/**
 * Base pragma parser.
 */
public class PragmaParser
{
    /** XobjectFile of target AST */
    private XobjectFile xobjFile;
    
    /** A stack of environments. */
    protected Stack<XobjList> stackEnv = new Stack<XobjList>();

    public PragmaParser(XobjectFile xobjFile)
    {
        this.xobjFile = xobjFile;
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

    public XobjectFile getXobjectFile()
    {
        return xobjFile;
    }
    
    public Xobject parse(Xobject x) throws XmException
    {
        return x;
    }
    
    public boolean isPrePostPair(Xobject prefix, Xobject postfix)
    {
        return false;
    }
    
    public void completePragmaEnd(Xobject prefix, Xobject body)
    {
      return;
    }
    
    public XobjArgs getAbbrevPostfix(XobjArgs prefixArgs)
    {
        return null;
    }
    
    public void mergeStartAndPostfixArgs(Xobject start, Xobject postfix)
    {
      return;
    }
}
