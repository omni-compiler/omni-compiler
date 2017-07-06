package xcodeml.c.decompile;
import java.util.ArrayList;
import java.util.List;
import xcodeml.util.XmException;
import xcodeml.c.obj.XcNode;
import xcodeml.c.util.XmcWriter;

/**
 * Internal object represents functionCall.
 */
public class XcFuncCallObj extends XcObj implements XcExprObj
{
    private XcExprObj _addrExpr;
    private List<XcExprObj> _argExprList = new ArrayList<XcExprObj>();

    /**
     * Creates XcFuncCallObj.
     */
    public XcFuncCallObj()
    {
    }

    /**
     * Sets an address of function.
     */
    public void setAddrExpr(XcExprObj addrExpr)
    {
        _addrExpr = addrExpr;
    }

    /**
     * Adds a function call argument.
     */
    public void addArg(XcExprObj arg)
    {
        _argExprList.add(arg);
    }

    @Override
    public void addChild(XcNode child)
    {
        if(child instanceof XcExprObj) {
            if(_addrExpr == null)
                _addrExpr = (XcExprObj)child;
            else
                _argExprList.add((XcExprObj)child);
        } else
            throw new IllegalArgumentException(child.getClass().getName());
    }

    @Override
    public void checkChild()
    {
        if(_addrExpr == null)
            throw new IllegalArgumentException("no function address");
    }

    @Override
    public XcNode[] getChild()
    {
        List<XcNode> nodeList = new ArrayList<XcNode>();
        nodeList.add(_addrExpr);
        nodeList.addAll(_argExprList);

        return nodeList.toArray(new XcNode[nodeList.size()]);
    }

    @Override
    public final void setChild(int index, XcNode child)
    {
        switch(index) {
        case 0:
            _addrExpr = (XcExprObj)child;
            break;
        default:
            if(index - 1 < _argExprList.size())
                _argExprList.set(index - 1, (XcExprObj)child);
            else
                throw new IllegalArgumentException(index + ":" + child.getClass().getName());
        }
    }
    
    @Override
    public void appendCode(XmcWriter w) throws XmException
    {
        w.addBraceIfNeeded(_addrExpr).add("(");
        for(int i = 0; i < _argExprList.size(); ++i) {
            if(i > 0)
                w.add(",");
            w.addSpc(_argExprList.get(i));
        }
        w.add(")");
    }
}
