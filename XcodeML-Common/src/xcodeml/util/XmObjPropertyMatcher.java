/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.util;

import xcodeml.XmObj;

/**
 * The matcher execute mathing about object's property.
 */
public class XmObjPropertyMatcher implements XmObjMatcher
{
    private String name;

    /**
     * Creates XmObjPropertyMatcher.
     *
     * @param name property name to match with.
     */
    public XmObjPropertyMatcher(String name)
    {
        this.name = name;
    }

    @Override
    public boolean match(XmObj n)
    {
        return n.hasProperty(name);
    }

}
