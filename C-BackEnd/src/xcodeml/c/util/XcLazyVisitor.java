package xcodeml.c.util;

import xcodeml.c.type.XcLazyEvalType;
import xcodeml.c.type.XcParamList;

/**
 * An interface provides 'lazyEnter.
 */
public interface XcLazyVisitor {
    /**
     * Sets the argument to a parent and enters his children which is <br>
     * XcodeML binding object not yet visited by a XcBindingVisitor.<br>
     * A lazyEnter function is used to lazy evaluate XcodeML binding objects<br>
     * those are not able to be evaluate at some timig but another timing<br>
     * such as the timing after evaluating while variables.
     * 
     * @param lazyType has XcodeML binding objects or DOM nodes these are not visited.
     */
    public void lazyEnter(XcLazyEvalType lazyType);

    public void pushParamListIdentTable(XcParamList paramList);

    public void popIdentTable();
}
