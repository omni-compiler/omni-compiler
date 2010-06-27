/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.XmObj;

/**
 * The matcher execute mathing about object's class.
 */
public class XmObjClassMatcher implements XmObjMatcher
{
    private Class<?> cls;

    /**
     * Creates XmObjClassMatcher
     *
     * @param name class name to match wich.
     */
    public XmObjClassMatcher(String name)
    {
        try {
            cls = Class.forName(name);
        } catch(Exception e) {
            cls = null;
        }
    }

    /**
     * Creates XmObjClassMatcher
     *
     * @param c class to match wich.
     */
    public XmObjClassMatcher(Class<?> c)
    {
        this.cls = c;
    }

    @Override
    public boolean match(XmObj n)
    {
        if(n == null) {
            return false;
        } else {
            return (n.getClass().equals(cls));
        }
    }

}
